import os
import shutil
import torch
from torch.nn import DataParallel, functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
import matplotlib.pyplot as plt
import numpy as np
from utils import parse_train_futefwi_args, create_training_dataset, train, test
from models import FuteFWI, Ablation_1, Ablation_2


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


def save_val_visualization(model, device, val_loader, save_path):
    """Only call from rank 0 when using DDP."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        data, target = batch
        pred = model(data.to(device, non_blocking=True)).detach().cpu().numpy()
        gt = target.numpy()
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


def setup_benchmark_seed(args):
    import random
    seed = getattr(args, 'seed', None)
    if seed is None:
        seed = int(os.environ.get('BENCHMARK_SEED', '42'))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if int(os.environ.get("LOCAL_RANK", "0")) <= 0:
        print(f'[Benchmark] Seed set to {seed}')


def _get_raw_model(model):
    """Extract raw model from DataParallel or DDP wrapper."""
    if isinstance(model, (DataParallel, DDP)):
        return model.module
    return model


def _model_save_path(args, name, dataset):
    suffix = "ND" if args.gaussian_noise else "D"
    return os.path.join(args.output, f"{name}_{dataset}_{suffix}.pt")


def _model_best_path(args, name, dataset):
    base, ext = os.path.splitext(_model_save_path(args, name, dataset))
    return base + "_best" + ext


if __name__ == "__main__":
    # parse args
    args = parse_train_futefwi_args()
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

    # create output dir (rank 0 only for DDP)
    if local_rank <= 0:
        os.makedirs(args.output, exist_ok=True)
    if use_ddp:
        torch.distributed.barrier()

    # load dataset
    if local_rank <= 0:
        print("Loading dataset...")
    dataset = f"{args.dataset}_{args.version}"
    if args.use_unified_loader:
        train_set, val_set = create_unified_training_dataset(args)
    else:
        train_set, val_set = create_training_dataset(dataset, args.gaussian_noise)
    if use_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
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
    if not args.ablation:
        model = FuteFWI(hidden_size=args.hidden_size, num_layers=args.layers, num_heads=args.heads)
        name = args.name
    elif args.ablation == "sfe":
        model = Ablation_1(hidden_size=args.hidden_size, num_layers=args.layers, num_heads=args.heads)
        name = f"{args.name}_ablation1"
    elif args.ablation == "tm":
        model = Ablation_2()
        name = f"{args.name}_ablation2"
    else:
        raise RuntimeError("Unexpected ablation type.")

    # 公平模式：Transformer 必须降维，否则计算量爆炸
    if os.environ.get('BENCHMARK_FAIR_MODE') == '1':
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from benchmark.fair_wrapper import FairWrapper
        if local_rank <= 0:
            print('🔥 [FairMode] Wrapping FuTE-FWI (256 -> 70 -> 256)')
        model = FairWrapper(model, target_size=(256, 256), internal_size=(70, 70))

    if use_ddp:
        if getattr(args, "sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[local_rank])
    elif args.device == "cuda":
        model = DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    # set optimizer
    loss = F.l1_loss
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # training（按 val loss 选最佳，最终 eval 用 best 而非最后一轮）
    vis_dir = os.path.join(args.output, "visualizations")
    train_loss, val_loss = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs = range(1, args.epochs + 1)
    for epoch in epochs:
        if use_ddp:
            train_sampler.set_epoch(epoch - 1)
        train_loss.append(train(model, device, train_loader, loss, optimizer, epoch, args.epochs))
        epoch_val_loss = test(model, device, val_loader, loss)
        val_loss.append(epoch_val_loss)
        model_to_save = _get_raw_model(model)
        if local_rank <= 0:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                best_path = _model_best_path(args, name, dataset)
                torch.save(model_to_save.state_dict(), best_path)
                print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch} -> saved {best_path}")
            if epoch % args.save_interval == 0:
                ckpt_path = os.path.join(args.output, f"{name}_{dataset}_epoch{epoch}.pt")
                torch.save(model_to_save.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
            if epoch % args.vis_interval == 0:
                vis_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.png")
                save_val_visualization(model, device, val_loader, vis_path)
                print(f"Saved visualization: {vis_path}")

    # 训练结束：将 best 复制为最终 eval 权重（与 unified_benchmark_train 一致）
    if local_rank <= 0:
        save_path = _model_save_path(args, name, dataset)
        best_path = _model_best_path(args, name, dataset)
        if os.path.isfile(best_path):
            shutil.copy2(best_path, save_path)
            print(f"[Benchmark] Eval will use best epoch {best_epoch} (val_loss={best_val_loss:.6f}) -> {save_path}")
            with open(os.path.join(args.output, "best_epoch.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_epoch={best_epoch}\nbest_val_loss={best_val_loss:.6f}\n")
        else:
            model_to_save = _get_raw_model(model)
            torch.save(model_to_save.state_dict(), save_path)
            print(f"Save the model to: {save_path}")
