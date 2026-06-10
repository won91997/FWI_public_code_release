"""Utility functions for training and testing"""

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from einops import rearrange
from tqdm.auto import tqdm
from pytorch_msssim import ssim, ms_ssim
import lpips
from skimage import feature
from skimage.metrics import hausdorff_distance as _hd
import time
import os
from utils.dataset import Dataset, LargeDataset, TestDataset

vel_families = ["FlatVel_A", "FlatVel_B", "CurveVel_A", "CurveVel_B"]
fault_familyies = ["FlatFault_A", "FlatFault_B", "CurveFault_A", "CurveFault_B"]


def create_training_dataset(dataset, gaussian_noise, memmap=True):
    """Create datasets according to different dataset family.

    Memory mapping (LargeDataset) is applied for the sake of the limited RAM. Note that adding gaussian noise is not
    implemented in LargeDataset.

    For who has enough memory capacity, set `memmap=False` to accelerate training.
    """
    if dataset in vel_families:
        train_range = range(0, 48)
        val_range = range(48, 60)
    elif dataset in fault_familyies:
        train_range = range(0, 96)
        val_range = range(96, 108)
    else:
        raise NotImplementedError("Unsupport dataset")
    if memmap:
        if gaussian_noise:
            train_set = LargeDataset(os.path.join("data", f"{dataset}_ND"), list(train_range))
            val_set = LargeDataset(os.path.join("data", f"{dataset}_ND"), list(val_range))
        else:
            train_set = LargeDataset(os.path.join("data", f"{dataset}_D"), list(train_range))
            val_set = LargeDataset(os.path.join("data", f"{dataset}_D"), list(val_range))
    else:
        if gaussian_noise:
            train_set = Dataset(os.path.join("data", f"{dataset}_ND"), list(train_range))
            val_set = Dataset(os.path.join("data", f"{dataset}_ND"), list(val_range))
        else:
            train_set = Dataset(os.path.join("data", f"{dataset}_D"), list(train_range))
            val_set = Dataset(os.path.join("data", f"{dataset}_D"), list(val_range))
    return train_set, val_set


def create_testing_dataset(dataset, gaussian_noise):
    if dataset in vel_families:
        val_range = range(48, 60)
    elif dataset in fault_familyies:
        val_range = range(96, 108)
    else:
        raise NotImplementedError("Unsupport dataset")

    if gaussian_noise:
        dataset = TestDataset(os.path.join("data", f"{dataset}_ND"), list(val_range))
    else:
        dataset = TestDataset(os.path.join("data", f"{dataset}_D"), list(val_range))
    return dataset


def train(model, device, dataloader, loss_fn, optimizer, epoch, epochs):
    """Train function (supports DDP: only rank 0 shows tqdm/print)."""
    model.train()
    batch_num = len(dataloader)
    train_loss = 0
    _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _disable_tqdm = _rank != 0
    for data, target in tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", ncols=70, disable=_disable_tqdm):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        l = loss_fn(output, target)
        l.backward()
        optimizer.step()
        train_loss += l.item()
    train_loss /= batch_num
    if _rank == 0:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{now}] Train set: Average loss: {train_loss:.6f}")
    return train_loss


def test(model, device, dataloader, loss_fn):
    """Test function (supports DDP: only rank 0 shows tqdm/print)."""
    model.eval()
    test_loss = 0
    _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _disable_tqdm = _rank != 0
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Testing", ncols=70, disable=_disable_tqdm):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
    test_loss /= len(dataloader)
    if _rank == 0:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{now}] Test set: Average loss: {test_loss:.6f}\n")
    return test_loss


def train_gan(
    model,
    model_d,
    device,
    dataloader,
    loss_g,
    loss_d,
    optimizer_g,
    optimizer_d,
    epoch,
    epochs,
    update_interval,
    grad_accum_steps=1,
):
    """Train function for VelocityGAN (supports DDP)"""
    model.train()
    train_loss = 0
    _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _disable_tqdm = _rank != 0
    for batch_idx, (data, label) in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", ncols=70, disable=_disable_tqdm)):
        data, label = data.to(device), label.to(device)
        optimizer_d.zero_grad()
        output = model(data)
        # detach output for D loss to avoid inplace/graph corruption when G and D share tensors (DDP-safe)
        # DDP 下用 model_d.module 计算 gradient penalty，避免 create_graph=True 时的 inplace 错误
        model_for_gp = getattr(model_d, "module", model_d)
        ld = loss_d(label, output.detach(), model_d, model_for_gp=model_for_gp)[0]
        ld.backward()
        optimizer_d.step()
        train_loss += ld.item()

        if (batch_idx + 1) % update_interval == 0:
            optimizer_g.zero_grad()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # 微批次：将 G 的 forward+backward 拆成多步，降低显存峰值（effective batch 不变）
            batch_sz = data.size(0)
            micro_sz = max(1, batch_sz // grad_accum_steps)
            for i in range(0, batch_sz, micro_sz):
                sub_data = data[i : i + micro_sz]
                sub_label = label[i : i + micro_sz]
                pred = model(sub_data)
                lg = loss_g(pred, sub_label, model_d) / grad_accum_steps
                lg.backward()
            optimizer_g.step()

    train_loss /= len(dataloader)
    if _rank == 0:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{now}] Train set: Average discriminator loss: {train_loss:.6f}")
    return train_loss


def test_gan(model, device, dataloader, loss_fn):
    """Test function for VelocityGAN (supports DDP)"""
    model.eval()
    test_loss = 0
    _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _disable_tqdm = _rank != 0
    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Testing", ncols=70, disable=_disable_tqdm):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)
            test_loss += loss_fn(output, label).item()
    test_loss /= len(dataloader)
    if _rank == 0:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{now}] Test set: Average generator loss: {test_loss:.6f}\n")
    return test_loss


def evaluate(dataloader, model, device):
    mae, mse, ssim_loss, mssim_loss, ps = 0, 0, 0, 0, 0
    total_samples = len(dataloader.dataset)
    lpips_loss = lpips.LPIPS(net="alex", verbose=False)
    if device == torch.device("cuda"):
        lpips_loss = lpips_loss.cuda()
    for data, label, _ in tqdm(dataloader, desc="Evaluating", ncols=60):
        batch_size = label.size(0)
        data, label = data.to(device), label.to(device)
        output = model(data)
        mae += torch.mean(torch.abs(output - label) * batch_size).item()
        mse += torch.mean((output - label) ** 2 * batch_size).item()
        ssim_loss += torch.sum(
            ssim(output / 2 + 0.5, label / 2 + 0.5, data_range=1, size_average=False)
        )  # [-1, 1] -> [0, 1]
        mssim_loss += torch.sum(
            ms_ssim(output / 2 + 0.5, label / 2 + 0.5, data_range=1, size_average=False, win_size=3)
        )  # [-1, 1] -> [0, 1]
        ps += torch.sum(lpips_loss(output, label))
    mae /= total_samples
    mse /= total_samples
    ssim_loss /= total_samples
    mssim_loss /= total_samples
    ps /= total_samples
    print(
        f"MAE: {mae:.4f}",
        f"MSE: {mse:.4f}",
        f"SSIM: {ssim_loss:.4f}",
        f"MS-SSIM: {mssim_loss:.4f}",
        f"LPIPS: {ps:.4f}",
        sep="\n",
    )


def evaluate_sample(dataloader, model, device):
    data, _, label = next(iter(dataloader))
    output = model(data.to(device))
    label_01 = (((label - label.min()) / (label.max() - label.min()) - 0.5) * 2).to(device)
    lpips_loss = lpips.LPIPS(net="alex", verbose=False)
    if device == torch.device("cuda"):
        lpips_loss = lpips_loss.cuda()
    mae = torch.mean(torch.abs(output - label_01))
    mse = torch.mean((output - label_01) ** 2)
    ssim_loss = ssim(output / 2 + 0.5, label_01 / 2 + 0.5, data_range=1, size_average=False)  # [-1, 1] -> [0, 1]
    mssim_loss = ms_ssim(
        output / 2 + 0.5, label_01 / 2 + 0.5, data_range=1, size_average=False, win_size=3
    )  # [-1, 1] -> [0, 1]
    ps = lpips_loss(output, label_01)
    hd, e_output, e_label = _hausdorff_distance(output, label_01)
    print(
        f"MAE: {mae.item():.4f}",
        f"MSE: {mse.item():.4f}",
        f"SSIM: {ssim_loss.item():.4f}",
        f"MS-SSIM: {mssim_loss.item():.4f}",
        f"LPIPS: {ps.item():.4f}",
        f"Hausdorff Distance: {hd.item():.4f}px",
        sep="\n",
    )

    # plot
    max_vals = torch.max(label).numpy()
    min_vals = torch.min(label).numpy()
    output = (output.detach().cpu().numpy() / 2 + 0.5) * (max_vals - min_vals) + min_vals
    output, label = np.squeeze(output), np.squeeze(label)

    fig, axs = plt.subplots(2, 2, figsize=(6, 12), sharex=True, sharey=True)
    extent = [0, 0.7, 1.4, 0]
    vmin = min(output.min(), min_vals)
    vmax = max(output.max(), max_vals)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    mappable = matplotlib.cm.ScalarMappable(norm)

    axs[0][0].imshow(label, norm=norm, extent=extent)
    axs[0][0].set_title(f"Ground Truth", {"fontsize": 12})
    axs[0][1].imshow(output, norm=norm, extent=extent)
    axs[0][1].set_title(f"Prediction", {"fontsize": 12})
    axs[1][0].imshow(e_label, cmap="gray", extent=extent)
    axs[1][0].set_title(f"Ground Truth (Edges)", {"fontsize": 12})
    axs[1][1].imshow(e_output, cmap="gray", extent=extent)
    axs[1][1].set_title(f"Prediction (Edges)", {"fontsize": 12})

    cb_ax = fig.add_axes([0.12, 0.51, 0.78, 0.01])
    fig.colorbar(mappable, cax=cb_ax, orientation="horizontal")
    fig.suptitle("Velocity Model and Interface Comparison", fontsize=16)
    plt.savefig(f"sample.png")


def plot_vmodel(dataloader, model, save_name, device):
    # Generate sample list
    data, _, label = next(iter(dataloader))
    output = torch.squeeze(model(data.to(device)).cpu()).numpy()
    label = torch.squeeze(label).numpy()

    # Renormalize
    max_vals = np.max(label)
    min_vals = np.min(label)
    output = (output / 2 + 0.5) * (max_vals - min_vals) + min_vals

    fig, axs = plt.subplots(1, 2, figsize=(6, 6.5))
    vmin = min(output.min(), min_vals)
    vmax = max(output.max(), max_vals)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    mappable = matplotlib.cm.ScalarMappable(norm)

    axs[0].imshow(output, norm=norm)
    axs[0].set_title(f"Prediction", {"fontsize": 12})
    axs[1].imshow(label, norm=norm)
    axs[1].set_title(f"Label", {"fontsize": 12})

    cb_ax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
    fig.colorbar(mappable, cax=cb_ax, orientation="horizontal")
    fig.suptitle(save_name, y=0.95, fontsize=20, fontweight=500)
    plt.show()
    plt.savefig(f"{save_name}.png")


def _hausdorff_distance(
    pred,
    label,
    sigma=0.8,
    low_threshold=0.02,
    high_threshold=0.08,
):
    """
    Calculate Hausdorff Distance between predicted velocity model and ground truth.
    """

    def _to_numpy(x):
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _normalize(x: np.ndarray) -> np.ndarray:
        """Normalize the ndarray to [0, 1]"""
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max == x_min:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min)

    pred = _normalize(_to_numpy(rearrange(pred, "b c h w -> (b c) h w")))
    label = _normalize(_to_numpy(rearrange(label, "b c h w -> (b c) h w")))

    B, _, _ = label.shape
    hd_arr = np.empty(B, dtype=float)
    for b in range(B):
        e_pred = feature.canny(pred[b], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        e_label = feature.canny(label[b], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        hd_arr[b] = _hd(e_label, e_pred)

    return hd_arr, e_pred, e_label


def _plot_loss(epoch, train_loss, val_loss, model, dataset):
    _, ax = plt.subplots()
    ax.plot(range(1, epoch + 1), train_loss, "b", label="Training loss")
    ax.plot(range(1, epoch + 1), val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(f"loss_{model}_{dataset}.png")
    plt.close()


def _median_filter(img, kernel_size=3):
    """img: b c h w"""
    pad_size = kernel_size // 2
    padded_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    unfolded_img = padded_img.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    unfolded_img = rearrange(unfolded_img, "b c h w k1 k2 -> b c h w (k1 k2)")

    median_filtered_img = unfolded_img.median(dim=-1)[0]
    return median_filtered_img
