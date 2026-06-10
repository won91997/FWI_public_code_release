import os
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from models import Generator, Discriminator
from utils import parse_train_velocitygan_args, create_training_dataset, train_gan, test_gan
from utils import Wasserstein_GP, UnionLoss


def save_val_visualization(model, device, val_loader, save_path):
    """Save pred vs GT comparison (consistent with unified_benchmark_train). Only call from rank 0 when using DDP."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        if len(batch) == 3:
            data, label = batch[0], batch[1]
        else:
            data, label = batch
        pred = model(data.to(device, non_blocking=True)).detach().cpu().numpy()
        gt = label.numpy() if hasattr(label, "numpy") else np.array(label)
        p = np.squeeze(pred[0, 0])
        g = np.squeeze(gt[0, 0])
        vmin = min(float(p.min()), float(g.min()))
        vmax = max(float(p.max()), float(g.max()))
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(p, cmap="jet", vmin=vmin, vmax=vmax)
        axs[0].set_title("Prediction")
        axs[0].axis("off")
        axs[1].imshow(g, cmap="jet", vmin=vmin, vmax=vmax)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


def create_unified_training_dataset(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from benchmark.unified_loader import UnifiedFWIDataset

    train_set = UnifiedFWIDataset(
        data_root=args.data_root,
        split="train",
        global_map_csv=args.global_map_csv,
        stats_json=args.stats_json,
        time_downsample=args.time_downsample,
        channel_mode=args.channel_mode,
        align_multiple=args.align_multiple,
        align_mode=args.align_mode,
        target_time=args.target_time,
        target_width=args.target_width,
        output_channel_dim=True,
    )
    val_set = UnifiedFWIDataset(
        data_root=args.data_root,
        split="val",
        global_map_csv=args.global_map_csv,
        stats_json=args.stats_json,
        time_downsample=args.time_downsample,
        channel_mode=args.channel_mode,
        align_multiple=args.align_multiple,
        align_mode=args.align_mode,
        target_time=args.target_time,
        target_width=args.target_width,
        output_channel_dim=True,
    )
    return train_set, val_set


def _get_raw_model(model):
    if isinstance(model, (DataParallel, DDP)):
        return model.module
    return model


def _model_save_path(args, dataset):
    suffix = "ND" if args.gaussian_noise else "D"
    return os.path.join(args.output, f"{args.name}_{dataset}_{suffix}.pt")


def _model_best_path(args, dataset):
    base, ext = os.path.splitext(_model_save_path(args, dataset))
    return base + "_best" + ext


def setup_benchmark_seed(args):
    import random
    import numpy as np
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = int(os.environ.get("BENCHMARK_SEED", "42"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Benchmark] Seed set to {seed}")


if __name__ == "__main__":
    args = parse_train_velocitygan_args()
    setup_benchmark_seed(args)

    # DDP: init when launched via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0
    if use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.barrier()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output, exist_ok=True)

    # load dataset
    if local_rank <= 0:
        print("Loading dataset...")
    dataset = f"{args.dataset}_{args.version}"
    output_size = (getattr(args, "output_height", 256), getattr(args, "output_width", 256))
    if args.use_unified_loader:
        train_set, val_set = create_unified_training_dataset(args)
    else:
        train_set, val_set = create_training_dataset(dataset, args.gaussian_noise)

    if use_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    # create model
    if local_rank <= 0:
        print("Using device:", device)
    model = Generator(output_size=output_size)
    model_d = Discriminator()

    # 公平模式下与 profile/eval 保持一致：入口降到 70，再上采样回 256。
    if os.environ.get('BENCHMARK_FAIR_MODE') == '1':
        from benchmark.fair_wrapper import FairWrapper, FAIR_MODELS_CONFIG
        internal = FAIR_MODELS_CONFIG.get("VelocityGAN", 70)
        if local_rank <= 0:
            print(f'🔥 [FairMode] Wrapping VelocityGAN (256 -> {internal} -> 256)')
        model = FairWrapper(model, target_size=output_size, internal_size=(internal, internal))

    # 公平模式：SyncBatchNorm 与其他 CNN 保持一致，保证物理量级一致性
    if use_ddp and (getattr(args, "sync_bn", False) or os.environ.get('BENCHMARK_FAIR_MODE') == '1'):
        if local_rank <= 0:
            print('🔥 [FairMode] VelocityGAN: Converting to SyncBatchNorm for physics consistency')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_d)

    if use_ddp:
        model = DDP(model.to(device), device_ids=[local_rank])
        model_d = DDP(model_d.to(device), device_ids=[local_rank])
    elif args.device == "cuda":
        model = DataParallel(model)
        model_d = DataParallel(model_d)
        model.to(device)
        model_d.to(device)
    else:
        model.to(device)
        model_d.to(device)

    # set optimizer
    loss_g = UnionLoss(args.lambda_g1v, args.lambda_g2v)
    loss_d = Wasserstein_GP(device, args.lambda_gp)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr_g, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)

    # training（按 val loss 选最佳，最终 eval 用 best 而非最后一轮）
    vis_dir = os.path.join(args.output, "visualizations")
    vis_interval = getattr(args, "vis_interval", 5)
    train_loss, val_loss = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs = range(1, args.epochs + 1)
    for epoch in epochs:
        if use_ddp:
            train_sampler.set_epoch(epoch - 1)
        train_loss.append(
            train_gan(
                model,
                model_d,
                device,
                train_loader,
                loss_g,
                loss_d,
                optimizer_g,
                optimizer_d,
                epoch,
                args.epochs,
                args.update_interval,
                grad_accum_steps=getattr(args, "grad_accum_steps", 1),
            )
        )
        epoch_val_loss = test_gan(model, device, val_loader, loss_g)
        val_loss.append(epoch_val_loss)

        if local_rank <= 0 and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_path = _model_best_path(args, dataset)
            torch.save(_get_raw_model(model).state_dict(), best_path)
            print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch} -> saved {best_path}")

        # 每 vis_interval 轮导出 pred vs GT 对比图（与其他模型一致）
        if (not use_ddp or local_rank == 0) and (epoch % vis_interval == 0 or epoch == args.epochs):
            model_to_vis = _get_raw_model(model)
            vis_path = os.path.join(vis_dir, "epoch_{:04d}.png".format(epoch))
            save_val_visualization(model_to_vis, device, val_loader, vis_path)
            if local_rank <= 0:
                print(f"Saved vis: {vis_path}")

    # 训练结束：将 best 复制为最终 eval 权重（与 unified_benchmark_train 一致）
    if not use_ddp or local_rank == 0:
        save_path = _model_save_path(args, dataset)
        best_path = _model_best_path(args, dataset)
        if os.path.isfile(best_path):
            shutil.copy2(best_path, save_path)
            print(f"[Benchmark] Eval will use best epoch {best_epoch} (val_loss={best_val_loss:.6f}) -> {save_path}")
            with open(os.path.join(args.output, "best_epoch.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_epoch={best_epoch}\nbest_val_loss={best_val_loss:.6f}\n")
        else:
            torch.save(_get_raw_model(model).state_dict(), save_path)
            print(f"Saved: {save_path}")

    if use_ddp:
        torch.distributed.destroy_process_group()
