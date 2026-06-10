import os
import torch
from torch.nn import DataParallel, functional as F
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import numpy as np
from utils import parse_train_inversionnet_args, create_training_dataset, train, test
from models import InversionNet


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
    model.eval()
    with torch.no_grad():
        data, target = next(iter(val_loader))
        pred = model(data.to(device)).detach().cpu().numpy()
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
    print(f'[Benchmark] Seed set to {seed}')


if __name__ == "__main__":
    # parse args
    args = parse_train_inversionnet_args()
    setup_benchmark_seed(args)

    # create output dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load dataset
    print("Loading dataset...")
    dataset = f"{args.dataset}_{args.version}"
    if args.use_unified_loader:
        train_set, val_set = create_unified_training_dataset(args)
    else:
        train_set, val_set = create_training_dataset(dataset, args.gaussian_noise)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    # create model
    device = torch.device(args.device)
    print("Using device:", device)
    output_size = (getattr(args, 'output_height', 256), getattr(args, 'output_width', 256))
    model = InversionNet(output_size=output_size)
    if args.device == "cuda":
        model = DataParallel(model)
    model = model.to(device)

    # set optimizer
    loss = F.l1_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    vis_dir = os.path.join(args.output, "visualizations")
    train_loss, val_loss = [], []
    epochs = range(1, args.epochs + 1)
    for epoch in epochs:
        train_loss.append(train(model, device, train_loader, loss, optimizer, epoch, args.epochs))
        val_loss.append(test(model, device, val_loader, loss))
        model_to_save = model.module if isinstance(model, DataParallel) else model
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output, f"{args.name}_{dataset}_epoch{epoch}.pt")
            torch.save(model_to_save.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
        if epoch % args.vis_interval == 0:
            vis_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.png")
            save_val_visualization(model, device, val_loader, vis_path)
            print(f"Saved visualization: {vis_path}")

    # save model
    model_to_save = model.module if isinstance(model, DataParallel) else model
    if args.gaussian_noise:
        save_path = os.path.join(args.output, f"{args.name}_{dataset}_ND.pt")
    else:
        save_path = os.path.join(args.output, f"{args.name}_{dataset}_D.pt")
    torch.save(model_to_save.state_dict(), save_path)
