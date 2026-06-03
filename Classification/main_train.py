import os
import time

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

from trainer import train, validate
from utils import (
    NormalizeByChannelMeanStd,
    save_checkpoint,
    setup_model_dataset,
    setup_seed,
)

best_sa = 0


def _resolve_loaders(args):
    if args.dataset == "imagenet":
        # 3-tuple: (model, train_loader, val_loader)
        args.class_to_replace = None
        out = setup_model_dataset(args)
        if len(out) != 3:
            raise RuntimeError(
                f"setup_model_dataset returned {len(out)} values for "
                f"--dataset imagenet, expected 3."
            )
        model, train_loader, val_loader = out
        return model, train_loader, val_loader

    if args.dataset in ("celeba", "imagenet100"):
        out = setup_model_dataset(args)
        if len(out) != 6:
            raise RuntimeError(
                f"setup_model_dataset returned {len(out)} values for "
                f"--dataset {args.dataset}, expected 6."
            )
        model, train_loader, val_loader, _test, _forget, _retain = out
        return model, train_loader, val_loader

    out = setup_model_dataset(args)
    if len(out) != 5:
        raise RuntimeError(
            f"setup_model_dataset returned {len(out)} values for "
            f"--dataset {args.dataset}, expected 5."
        )
    model, train_loader, val_loader, _test, _marked = out
    return model, train_loader, val_loader


def main():
    global best_sa
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    model, train_loader, val_loader = _resolve_loaders(args)
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.imagenet_arch:
        warmup = max(1, int(args.warmup or 1))

        def _lambda(cur_iter):
            if cur_iter < warmup:
                return (cur_iter + 1) / warmup
            return 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * ((cur_iter - warmup) / max(1, args.epochs - warmup))
                )
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1
        )

    start_epoch = 0
    state = 0
    all_result = {"train_ta": [], "val_ta": [], "test_ta": []}

    if args.resume and getattr(args, "checkpoint", None):
        print(f"resume from checkpoint {args.checkpoint}")
        checkpoint = torch.load(
            args.checkpoint,
            map_location=torch.device(f"cuda:{int(args.gpu)}"),
        )
        best_sa = checkpoint.get("best_sa", 0)
        start_epoch = checkpoint.get("epoch", 0)
        all_result = checkpoint.get("result", all_result)

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"loading from epoch: {start_epoch}, best_sa={best_sa}")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(
            "Epoch #{}, Learning rate: {}".format(
                epoch,
                optimizer.state_dict()["param_groups"][0]["lr"],
            )
        )

        acc = train(train_loader, model, criterion, optimizer, epoch, args)
        tacc = validate(val_loader, model, criterion, args)
        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)

        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_SA_best=is_best_sa,
            pruning=state,
            save_path=args.save_dir,
        )
        print(f"one epoch duration:{time.time() - start_time}")

    try:
        plt.plot(all_result["train_ta"], label="train_acc")
        plt.plot(all_result["val_ta"], label="val_acc")
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, f"{state}net_train.png"))
        plt.close()
    except Exception as exc:  # never fail training over a plot
        print(f"[warn] failed to draw curve: {exc}")

    print("Performance on the validation set (final):")
    final_tacc = validate(val_loader, model, criterion, args)
    print(f"final val acc = {final_tacc:.3f}")

    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = int(np.argmax(np.array(all_result["val_ta"])))
        print(
            "* best SA = {:.3f}, Epoch = {}".format(
                all_result["val_ta"][val_pick_best_epoch],
                val_pick_best_epoch + 1,
            )
        )


if __name__ == "__main__":
    main()
