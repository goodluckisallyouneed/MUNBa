import gc
import time

import torch
import torch.nn as nn
from torch.autograd import grad
from tqdm import tqdm

import utils

# Re-use the head-selection / forward helpers from IMU so IU and IMU operate
# on *exactly* the same parameter subspace and CLIP forward path.
from unlearn.IMU import (
    _clip_logits,
    _freeze_to_head,
    _select_head_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flat_grad(loss, params):
    """Flatten autograd.grad output over `params` into a single 1-D tensor."""
    grads = grad(loss, params, retain_graph=False, create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


def _apply_perturb(params, v):
    """In-place: param += v (chunked back into each param's shape)."""
    curr = 0
    with torch.no_grad():
        for p in params:
            length = p.numel()
            chunk = v[curr:curr + length].view_as(p).to(p.dtype).to(p.device)
            p.data.add_(chunk)
            curr += length
    assert curr == v.numel(), (
        f"perturbation size mismatch: consumed {curr}, vector length {v.numel()}"
    )


def _accumulate_mean_grad(model, loader, texts, logit_scale, mode, params,
                          device, criterion, desc):
    """Sum of (per-batch loss-grad * batch_size) over `loader`, returns
    (sum_grad [P], total_samples)."""
    flat_dim = sum(p.numel() for p in params)
    grad_sum = torch.zeros(flat_dim, device=device)
    total = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        images, targets = batch[0], batch[1]
        images = images.to(device)
        targets = targets.to(device).long()
        real_num = images.size(0)

        logits = _clip_logits(model, images, texts, logit_scale, mode)
        loss = criterion(logits, targets)
        flat = _flat_grad(loss, params).detach()
        grad_sum = grad_sum + flat * real_num
        total += real_num
    if total == 0:
        raise RuntimeError(f"IU: empty loader for '{desc}'")
    return grad_sum, total


def _woodfisher(model, retain_loader, texts, logit_scale, mode, params,
                device, criterion, v, N, max_samples):
    """Streaming Woodfisher rank-1 approximation of H^{-1} v.

    Iterates ``retain_loader`` (with batch_size == 1, see caller) and updates
    k_vec, o_vec by the recursion:

        tmp     = <o_vec, g_i>
        k_vec  -= (<k_vec, g_i> / (N + tmp)) * o_vec
        o_vec  -= (tmp / (N + tmp)) * o_vec

    where g_i is the per-sample gradient. Stops after ``max_samples``
    iterations (mirrors the original `if idx > N: return` guard).
    """
    model.eval()
    k_vec = v.clone()
    o_vec = None

    pbar = tqdm(retain_loader, desc="IU: Woodfisher", leave=False)
    for idx, batch in enumerate(pbar):
        images, targets = batch[0], batch[1]
        images = images.to(device)
        targets = targets.to(device).long()

        logits = _clip_logits(model, images, texts, logit_scale, mode)
        loss = criterion(logits, targets)
        sample_grad = _flat_grad(loss, params).detach()

        with torch.no_grad():
            if o_vec is None:
                o_vec = sample_grad.clone()
            else:
                tmp = torch.dot(o_vec, sample_grad)
                denom = N + tmp
                k_vec -= (torch.dot(k_vec, sample_grad) / denom) * o_vec
                o_vec -= (tmp / denom) * o_vec

        if idx + 1 >= max_samples:
            break

    return k_vec


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def IU(texts, data_loaders, model, args, class_name):
    """IU / Wfisher on CLIP.

    Head-only by construction so that IU and IMU operate on the same
    parameter subspace (see module docstring for rationale).
    """
    forget_loader_in = data_loaders["forget"]
    retain_loader_in = data_loaders["retain"]
    device = torch.device(
        f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    )

    # CLIP defaults to fp16; autograd.grad on small CE is unstable. Cast to
    # fp32 for IU only (matches IMU).
    model.float()

    # Restrict the trainable / differentiated parameter set to the CLIP head.
    _freeze_to_head(model, args.mode)
    head_named, head_params = _select_head_params(model, args.mode)
    n_head = sum(p.numel() for p in head_params)
    print(
        f"[IU] mode={args.mode}, head params: "
        f"{[n for n, _ in head_named]}, total {n_head} elements"
    )

    logit_scale = 100
    criterion = nn.CrossEntropyLoss()

    # Re-wrap loaders. We use:
    #   * forget set: batch_size = args.batch_size (mean-grad, fast)
    #   * retain set, mean-grad pass: batch_size = args.batch_size
    #   * retain set, Woodfisher pass: batch_size = 1 (per-sample grads)
    forget_grad_loader = torch.utils.data.DataLoader(
        forget_loader_in.dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader_in.dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    retain_persample_loader = torch.utils.data.DataLoader(
        retain_loader_in.dataset, batch_size=1,
        shuffle=False, num_workers=0,
    )

    # ---- Stage 1: forget mean gradient ---------------------------------------
    t0 = time.time()
    forget_grad, total_f = _accumulate_mean_grad(
        model, forget_grad_loader, texts, logit_scale, args.mode,
        head_params, device, criterion, desc="IU: forget grad",
    )

    # ---- Stage 2: retain mean gradient ---------------------------------------
    retain_grad, total_r = _accumulate_mean_grad(
        model, retain_grad_loader, texts, logit_scale, args.mode,
        head_params, device, criterion, desc="IU: retain grad",
    )

    # Same scaling as the original Wfisher implementation:
    #   forget_grad <- forget_grad / (total_f + total_r)
    #   retain_grad <- retain_grad * total_f / ((total_f + total_r) * total_r)
    denom = float(total_f + total_r)
    forget_grad = forget_grad / denom
    retain_grad = retain_grad * (float(total_f) / (denom * float(total_r)))

    v = forget_grad - retain_grad
    print(
        f"[IU] mean-grad pass done in {time.time() - t0:.1f}s; "
        f"|v|2={v.norm().item():.4e}, total_f={total_f}, total_r={total_r}"
    )

    # ---- Stage 3: Woodfisher H^{-1} v on retain set --------------------------
    N = float(getattr(args, "iu_N", 1000))
    # Cap the per-sample retain pass to keep IU tractable on ImageNet-100.
    user_max = int(getattr(args, "iu_max_retain_samples", 0))
    available = len(retain_persample_loader)
    if user_max > 0:
        max_samples = min(user_max, available)
    else:
        # Default mirrors the CIFAR setting in the reference code.
        max_samples = min(1000, available)
    print(
        f"[IU] Woodfisher: N={N}, max_samples={max_samples} "
        f"(retain set has {available} samples)"
    )

    t1 = time.time()
    perturb = _woodfisher(
        model, retain_persample_loader, texts, logit_scale, args.mode,
        head_params, device, criterion, v=v, N=N, max_samples=max_samples,
    )
    print(
        f"[IU] Woodfisher done in {time.time() - t1:.1f}s; "
        f"|H^-1 v|2={perturb.norm().item():.4e}"
    )

    # ---- Stage 4: closed-form perturbation -----------------------------------
    alpha = float(args.alpha)
    print(f"[IU] applying perturbation: theta <- theta + {alpha} * H^-1 v")
    _apply_perturb(head_params, alpha * perturb)

    # House-keeping (matches IMU.py).
    torch.cuda.empty_cache()
    gc.collect()

    print("[IU] done.")
