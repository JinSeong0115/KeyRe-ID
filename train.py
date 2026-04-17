import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

from evaluation import test
from heatmap_loader import heatmap_dataloader
from keyreid import KeyReID
from losses import make_loss
from utils import AverageMeter
from utils import optimizer as build_optimizer
from utils import scheduler as build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="KeyRe-ID")
    parser.add_argument("--dataset_name", default="MARS", type=str, help="The name of the dataset")
    parser.add_argument(
        "--ViT_path",
        default="./weights/jx_vit_base_p16_224-80ecf9dd.pth",
        required=True,
        type=str,
        help="Path to the pre-trained Vision Transformer model",
    )
    parser.add_argument(
        "--dataset_root",
        default="./data",
        required=True,
        type=str,
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Directory to save weights, logs, and evaluation results",
    )
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_interval", default=10, type=int)
    parser.add_argument("--log_interval", default=200, type=int)
    parser.add_argument("--print_interval", default=400, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--eval_pool", default="avg", choices=["avg", "max"])
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def resolve_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(args, num_classes, camera_num, device):
    model = KeyReID(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.ViT_path)
    print("Running load_param")
    model.load_param(args.ViT_path)
    return model.to(device)


def compute_accuracy(score, pid):
    logits = score[0] if isinstance(score, list) else score
    return (logits.max(1)[1] == pid).float().mean()


def write_training_log(path, epoch, iteration, loss, acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as log_file:
        log_file.write(
            f"Epoch {epoch}, Iteration {iteration}, Loss: {loss:.6f}, Acc: {acc:.3f}\n"
        )


def train_one_epoch(
    model,
    train_loader,
    loss_func,
    center_criterion,
    optimizer,
    optimizer_center,
    lr_scheduler,
    scaler,
    ema,
    device,
    epoch,
    args,
    loss_history,
    loss_log_path,
):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    lr_scheduler.step(epoch)
    model.train()

    for iteration, (imgs, heatmaps, pid, target_cam, erasing_labels) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        optimizer_center.zero_grad()

        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        pid = pid.to(device)
        target_cam = target_cam.to(device).view(-1)
        erasing_labels = erasing_labels.to(device)

        with amp.autocast(enabled=device.type == "cuda"):
            score, feat, attention_values = model(imgs, heatmaps, pid, cam_label=target_cam)
            attention_loss = (attention_values * erasing_labels).sum(dim=1).mean()
            loss_id, center = loss_func(score, feat, pid)
            loss = loss_id + 0.0005 * center + attention_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        for param in center_criterion.parameters():
            if param.grad is not None:
                param.grad.data *= 1.0 / 0.0005
        scaler.step(optimizer_center)
        scaler.update()

        acc = compute_accuracy(score, pid)
        loss_meter.update(loss.item(), imgs.shape[0])
        acc_meter.update(acc, 1)
        loss_history.append(loss.item())

        if iteration % args.log_interval == 0:
            write_training_log(loss_log_path, epoch, iteration, loss.item(), acc_meter.avg)

        if device.type == "cuda":
            torch.cuda.synchronize()

        if iteration % args.print_interval == 0:
            lr = lr_scheduler._get_lr(epoch)[0]
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch,
                    iteration,
                    len(train_loader),
                    loss_meter.avg,
                    acc_meter.avg,
                    lr,
                )
            )


def evaluate_and_save(model, q_val_set, g_val_set, dataset_name, epoch, best_rank1, best_map, args, use_gpu):
    cmc, cmc5, mean_ap = test(model, q_val_set, g_val_set, pool=args.eval_pool, use_gpu=use_gpu)
    print("CMC: %.4f, R-5: %.4f, mAP : %.4f" % (cmc, cmc5, mean_ap))

    eval_dir = os.path.join(args.output_dir, "evaluate")
    weight_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    with open(os.path.join(eval_dir, "matrix_best.txt"), "a") as log_file:
        log_file.write(f"Epoch {epoch}: CMC = {cmc:.4f}, R-5 = {cmc5:.4f}, mAP = {mean_ap:.4f}\n")

    if best_rank1 < cmc:
        best_rank1 = cmc
        torch.save(model.state_dict(), os.path.join(weight_dir, f"{dataset_name}best_CMC.pth"))
    if best_map < mean_ap:
        best_map = mean_ap
        torch.save(model.state_dict(), os.path.join(weight_dir, f"{dataset_name}best_mAP.pth"))

    return best_rank1, best_map


def save_loss_plot(loss_history, loss_graph_path):
    os.makedirs(os.path.dirname(loss_graph_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(loss_graph_path)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_gpu = device.type == "cuda"

    heatmap_train_loader, _, num_classes, camera_num, _, q_val_set, g_val_set = heatmap_dataloader(
        args.dataset_name,
        args.dataset_root,
    )

    model = build_model(args, num_classes, camera_num, device)
    loss_func, center_criterion = make_loss(num_classes=num_classes, use_gpu=use_gpu)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer = build_optimizer(model)
    lr_scheduler = build_scheduler(optimizer)
    scaler = amp.GradScaler(enabled=use_gpu)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    best_rank1 = 0
    best_map = 0
    loss_history = []

    loss_dir = os.path.join(args.output_dir, "loss")
    loss_log_path = os.path.join(loss_dir, "loss_log_best.txt")
    loss_graph_path = os.path.join(loss_dir, "loss_plot_best.png")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_one_epoch(
            model=model,
            train_loader=heatmap_train_loader,
            loss_func=loss_func,
            center_criterion=center_criterion,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            args=args,
            loss_history=loss_history,
            loss_log_path=loss_log_path,
        )
        print(f"Epoch {epoch} finished in {time.time() - start_time:.1f}s")

        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            best_rank1, best_map = evaluate_and_save(
                model=model,
                q_val_set=q_val_set,
                g_val_set=g_val_set,
                dataset_name=args.dataset_name,
                epoch=epoch,
                best_rank1=best_rank1,
                best_map=best_map,
                args=args,
                use_gpu=use_gpu,
            )

    save_loss_plot(loss_history, loss_graph_path)
    print(f"Loss logs have been saved: {loss_log_path}")
    print(f"The loss graph has been saved: {loss_graph_path}")

    # Print final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (seed={args.seed})")
    print(f"  Best Rank-1: {best_rank1:.4f}")
    print(f"  Best mAP:    {best_map:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
