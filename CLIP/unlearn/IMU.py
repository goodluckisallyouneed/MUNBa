"""IMU (Influence-guided Machine Unlearning) for CLIP.

Adapted from the original implementation
(``IMU/Classification/unlearn/IMU.py``). Differences vs. the ResNet version:

* The "fc" layer no longer exists in CLIP. We use the joint-embedding
  *projection* layer + the LayerNorm right before it as the head:
    - ``mode == "text"``  : ``text_projection`` + ``ln_final.{weight,bias}``
    - ``mode == "image"`` : ``visual.proj`` + ``visual.ln_post.{weight,bias}``
    - ``mode == "all"``   : both
  This head is used **both** for influence/Fisher computation and as the
  trainable parameter set for the SGD unlearn step (so the influence signal
  and the update direction live in the same subspace).

* The forward pass mirrors ``unlearn/GA.py``: cosine similarity between
  CLIP image features and text features over the class prompts, scaled by
  ``logit_scale = 100``.

* Per-sample influence scores are computed with a streaming two-pass loop
  to avoid materializing the full [N, P] gradient matrix.
"""

import gc
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils


# ---------------------------------------------------------------------------
# Head selection
# ---------------------------------------------------------------------------
TEXT_HEAD_KEYS = {"text_projection", "ln_final.weight", "ln_final.bias"}
IMAGE_HEAD_KEYS = {"visual.proj", "visual.ln_post.weight", "visual.ln_post.bias"}


def _head_keys(mode):
    if mode == "text":
        return set(TEXT_HEAD_KEYS)
    if mode == "image":
        return set(IMAGE_HEAD_KEYS)
    if mode == "all":
        return TEXT_HEAD_KEYS | IMAGE_HEAD_KEYS
    raise ValueError(f"Unknown args.mode={mode!r}; expected text/image/all.")


def _select_head_params(model, mode):
    """Return (named_list, param_list) of head parameters under the given mode."""
    keys = _head_keys(mode)
    named = [(n, p) for n, p in model.named_parameters() if n in keys]
    found = {n for n, _ in named}
    missing = keys - found
    if missing:
        raise RuntimeError(
            f"IMU: expected head parameters not found in model: {sorted(missing)}. "
            f"Available top-level params include: "
            f"{sorted({n for n, _ in model.named_parameters()})[:20]} ..."
        )
    params = [p for _, p in named]
    return named, params


def _freeze_to_head(model, mode):
    """Freeze the whole CLIP, then unfreeze only the head parameters."""
    keys = _head_keys(mode)
    for n, p in model.named_parameters():
        p.requires_grad = (n in keys)


def l1_regularization(parameters):
    flat = [p.reshape(-1) for p in parameters]
    return torch.linalg.norm(torch.cat(flat), ord=1)


# ---------------------------------------------------------------------------
# CLIP forward + per-sample CE
# ---------------------------------------------------------------------------
def _clip_logits(model, images, texts, logit_scale, mode):
    """Compute CLIP cosine-similarity logits.

    During influence/Fisher computation we want gradients w.r.t. the head
    parameters only; samples without grad on the "other" tower are saved
    via ``torch.no_grad()``.
    """
    if mode == "text":
        with torch.no_grad():
            image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
    elif mode == "image":
        image_features = model.encode_image(images)
        with torch.no_grad():
            text_features = model.encode_text(texts)
    else:  # all
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return logit_scale * image_features @ text_features.t()


def _per_sample_ce(model, image, target, texts, logit_scale, mode):
    """Mean CE on a 1-sample mini-batch; suitable for autograd.grad."""
    logits = _clip_logits(model, image, texts, logit_scale, mode)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return torch.nn.functional.nll_loss(log_probs, target, reduction="mean")


def _flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


# ---------------------------------------------------------------------------
# Influence computation (streaming, no full G matrix)
# ---------------------------------------------------------------------------
def _iter_forget_samples(forget_loader, device):
    """Yield (image[1,3,H,W], target[1]) one sample at a time.

    We re-use the existing forget_loader's dataset to ensure transforms /
    indexing are consistent with downstream training.
    """
    dataset = forget_loader.dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        # Both OxfordPets.__getitem__ and Custom_Subset return (img, target).
        image, target = sample
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        target_tensor = torch.as_tensor(target).long().reshape(1)
        yield idx, image.unsqueeze(0).to(device), target_tensor.to(device)


def compute_fisher_diag(model, forget_loader, texts, logit_scale,
                        head_params, mode, device, eps):
    """Diagonal Fisher inverse over the forget set, head-only."""
    model.eval()
    fisher_diag = None
    count = 0
    for _, image, target in tqdm(
        list(_iter_forget_samples(forget_loader, device)),
        desc="IMU: Fisher diag", leave=False,
    ):
        loss = _per_sample_ce(model, image, target, texts, logit_scale, mode)
        flat = _flat_grad(loss, head_params).detach()
        sq = flat * flat
        fisher_diag = sq if fisher_diag is None else fisher_diag + sq
        count += 1
    if count == 0:
        raise RuntimeError("IMU: forget set is empty; cannot compute Fisher.")
    fisher_diag = fisher_diag / count + eps
    return 1.0 / fisher_diag


def compute_influences(model, forget_loader, texts, logit_scale,
                       head_params, mode, device, inv_fisher_diag):
    """Per-sample influences, streaming two-pass.

    Pass 1 accumulates ``grad_d2_total = (1/N) * sum_i g_i``.
    Pass 2 computes ``inf_i = -((g_i ⊙ inv_fisher) · grad_d2_total)``.
    """
    model.eval()

    # Pass 1: mean gradient
    grad_d2_total = None
    n = 0
    for _, image, target in tqdm(
        list(_iter_forget_samples(forget_loader, device)),
        desc="IMU: pass1 (mean grad)", leave=False,
    ):
        loss = _per_sample_ce(model, image, target, texts, logit_scale, mode)
        flat = _flat_grad(loss, head_params).detach()
        grad_d2_total = flat if grad_d2_total is None else grad_d2_total + flat
        n += 1
    grad_d2_total = grad_d2_total / n

    # Pass 2: per-sample influence (no [N, P] storage)
    influences = torch.empty(n, dtype=torch.float32)
    weighted_target = inv_fisher_diag * grad_d2_total  # [P]
    for i, (_, image, target) in enumerate(tqdm(
        list(_iter_forget_samples(forget_loader, device)),
        desc="IMU: pass2 (per-sample inf)", leave=False,
    )):
        loss = _per_sample_ce(model, image, target, texts, logit_scale, mode)
        flat = _flat_grad(loss, head_params).detach()
        influences[i] = -(flat @ weighted_target).item()
    return influences


# ---------------------------------------------------------------------------
# Influence-weighted dataset
# ---------------------------------------------------------------------------
class InfluenceWeightedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, weights):
        self.dataset = base_dataset
        # store as tensor for cheap indexing; one weight per item in base_dataset
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        assert len(self.weights) == len(self.dataset), (
            f"weights ({len(self.weights)}) and dataset ({len(self.dataset)}) "
            f"must have the same length."
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        if not torch.is_tensor(target):
            target = torch.as_tensor(target).long()
        return image, target, self.weights[index]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def IMU(texts, data_loaders, model, args, class_name):
    """IMU on CLIP. Head-only training (projection + last LayerNorm)."""
    forget_loader = data_loaders["forget"]
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")

    # CLIP fp16 + autograd.grad on small per-sample CE is unstable; cast to fp32
    # for the duration of IMU. This is local: we don't cast back, but other
    # methods would not run inside the same process anyway (one method per run).
    model.float()

    # Freeze everything, unfreeze only the head for both influence + update.
    _freeze_to_head(model, args.mode)
    head_named, head_params = _select_head_params(model, args.mode)

    n_head = sum(p.numel() for p in head_params)
    print(
        f"[IMU] mode={args.mode}, head params: "
        f"{[n for n, _ in head_named]}, total {n_head} elements"
    )

    logit_scale = 100
    eps = float(getattr(args, "imu_eps", 0.01))
    clip_q = float(getattr(args, "imu_clip_quantile", 0.93))
    top_data = float(getattr(args, "top_data", 1.0))

    # ---- Stage 1: Fisher diag inverse over forget set (head-only) ------------
    print("[IMU] computing diagonal Fisher inverse ...")
    inv_fisher_diag = compute_fisher_diag(
        model, forget_loader, texts, logit_scale,
        head_params, args.mode, device, eps,
    )

    # ---- Stage 2: per-sample influences (streaming two-pass) -----------------
    print("[IMU] computing per-sample influences ...")
    influences = compute_influences(
        model, forget_loader, texts, logit_scale,
        head_params, args.mode, device, inv_fisher_diag,
    )
    print(
        f"[IMU] influences: n={len(influences)}, "
        f"mean={influences.mean().item():.4e}, "
        f"min={influences.min().item():.4e}, "
        f"max={influences.max().item():.4e}, "
        f"#negative={(influences < 0).sum().item()}"
    )

    # ---- Stage 3: select negative-influence samples, top-k by |inf| ---------
    neg_mask = influences < 0
    neg_indices = torch.nonzero(neg_mask, as_tuple=False).reshape(-1)
    if len(neg_indices) == 0:
        print("[IMU] WARNING: no sample has negative influence; "
              "falling back to all forget samples ranked by |influence|.")
        neg_indices = torch.arange(len(influences))
        neg_influences = influences.clone()
    else:
        neg_influences = influences[neg_mask]

    abs_inf = neg_influences.abs()
    order = torch.argsort(abs_inf, descending=True)
    top_k = max(int(len(neg_influences) * top_data), 1)
    top_local = order[:top_k]
    selected_indices = neg_indices[top_local].tolist()
    selected_influences = neg_influences[top_local]

    print(f"[IMU] selecting top {top_data*100:.1f}% of negatives -> "
          f"{len(selected_indices)} / {len(forget_loader.dataset)} samples")

    # ---- Stage 4: build per-sample weights w = sqrt(|inf|), clip at q-quantile
    weights = torch.sqrt(selected_influences.abs())
    if len(weights) > 1:
        max_clip = torch.quantile(weights, clip_q)
        weights = torch.clamp(weights, max=max_clip)

    selected_subset = torch.utils.data.Subset(forget_loader.dataset, selected_indices)
    weighted_dataset = InfluenceWeightedDataset(selected_subset, weights)
    train_loader = torch.utils.data.DataLoader(
        weighted_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(f"[IMU] weighted training set: {len(train_loader.dataset)} samples")

    # ---- Stage 5: optimizer / scheduler --------------------------------------
    optimizer = torch.optim.SGD(
        head_params,
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1,
    )

    criterion_per_sample = nn.CrossEntropyLoss(reduction="none")

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    loader_len = len(train_loader)
    alpha = float(args.alpha)

    # ---- Stage 6: weighted gradient ascent on the head -----------------------
    for epoch in range(args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print(
            f"Epoch #{epoch}, Learning rate: "
            f"{optimizer.state_dict()['param_groups'][0]['lr']}"
        )

        start = time.time()
        for i, batch in enumerate(train_loader):
            images, targets, w = batch
            images = images.to(device)
            targets = targets.to(device).long()
            w = w.to(device).float()

            logits = _clip_logits(model, images, texts, logit_scale, args.mode)
            ce_per_sample = criterion_per_sample(logits, targets)
            # weighted gradient ASCENT on the (selected) forget set
            weighted_sum = w.sum().clamp_min(1e-8)
            loss = -(w * ce_per_sample).sum() / weighted_sum
            if alpha > 0:
                loss = loss + alpha * l1_regularization(head_params)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()

            with torch.no_grad():
                prec = utils.accuracy(logits, targets)[0]
                losses.update(loss.float().item(), images.size(0))
                top1.update(prec.item(), images.size(0))
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, loader_len, end - start,
                        loss=losses, top1=top1,
                    )
                )
                start = time.time()

        scheduler.step()
        print("one epoch duration:{}".format(time.time() - start_time))

    print("[IMU] done.")
