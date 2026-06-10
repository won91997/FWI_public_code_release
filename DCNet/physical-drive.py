from os.path import exists

import torch
import deepwave
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy.io as sio

# ##############################
# 参数配置（物理约束）
# ##############################
nx, ny = 70, 70  # 模型网格尺寸（70x70）
dx = 10.0  # 空间步长（米）
dt = 0.001  # 时间步长（秒）
nt = 1000  # 时间采样点数
peak_freq = 15.0  # 震源主频（Hz）
n_shots = 5  # 炮数
n_receivers = 70  # 每炮接收器数（70个）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成速度模型（真实模型含异常体，初始模型为平滑背景）
def create_velocity_model():
    velocity_path = "D:"  # 速度模型路径
    true_velocity = np.load(velocity_path)[0,0,:,:]
    # 生成初始速度模型（高斯滤波平滑处理）
    initial_velocity = gaussian_filter(true_velocity, sigma=5)
    initial_velocity = torch.tensor(initial_velocity, requires_grad=True)

    true_velocity = torch.tensor(true_velocity, requires_grad=True)
    return true_velocity, initial_velocity


v_true, v_init = create_velocity_model()
# 震源位置（使用网格索引）
src_locs = torch.zeros(n_shots, 1, 2, dtype=torch.long)
src_locs[:, 0, 0] = torch.linspace(0, nx-1, n_shots)
src_locs[:, 0, 1] = 0  # (5,1,2)

# 接收器位置（同理）
rec_locs = torch.zeros(n_shots, 70, 2, dtype=torch.long)
rec_locs[..., 0] = torch.linspace(0, nx-1, 70).repeat(n_shots, 1)
rec_locs[..., 1] = 0 #(5，70，2)


# ##############################
# 正演模拟（生成观测数据）
# ##############################
def forward_modeling(v_model):
    # 定义震源时间函数（雷克子波）
    source_amps = (deepwave.wavelets.ricker(peak_freq, nt,  dt, 1/peak_freq)
                   .unsqueeze(0).unsqueeze(0)
                   .repeat(n_shots, 1, 1))

    # 基于声波方程正演
    obs_data = deepwave.scalar(
        v_model,  # 速度模型
        dx, dt,
        source_amps,
        src_locs.repeat(1, 1, 1),
        rec_locs.repeat(1, 1, 1),
        pml_width=10  # PML吸收边界宽度
    )[-1].cpu()  # 接收器数据：[n_shots, nt, n_receivers]

    return obs_data
# 生成观测数据（真实模型）
obs_data = forward_modeling(v_true)  #【5，70，1000】

# ##############################
# 全波形反演主流程
# ##############################
# 初始化可训练参数（速度模型）
v_inv = v_init.clone().requires_grad_(True)
v_inv = torch.nn.Parameter(v_inv)
# 初始化优化器（Adam）
optimizer = torch.optim.Adam([v_inv], lr=2)
# 损失函数为MSE
criterion = torch.nn.MSELoss()
# 记录损失值
loss_history = []
epoch = 300
# 迭代优化
for epoch in range(epoch):
    optimizer.zero_grad()
    # 正演计算当前模型
    pred_data = forward_modeling(v_inv)

    # 计算L2数据残差
    loss = criterion(obs_data, pred_data)
    loss_history.append(loss.item())
    # 反向传播
    loss.backward(retain_graph=True)

    # 梯度平滑（在无梯度上下文中操作）
    with torch.no_grad():
        # 确保卷积核与模型在同一设备
        kernel = torch.ones(1, 1, 5, 5, device=v_inv.device) / 25
        grad_smooth = torch.nn.functional.conv2d(
            v_inv.grad.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=2
        ).squeeze()
        v_inv.grad.data.copy_(grad_smooth)

    # 参数更新
    optimizer.step()

    # 每 100 轮保存一次速度模型
    if (epoch + 1) % 100 == 0:
        plt.figure(figsize=(5, 5))
        plt.imshow(v_inv.detach().cpu().numpy(), cmap='viridis')
        plt.title(f'Predicted Velocity Model (Epoch {epoch + 1})')
        plt.colorbar()
        plt.savefig(f'pred_velocity_{epoch + 1}.png', dpi=300)
        plt.close()
    print(f'Epoch {epoch}, Loss: {loss.item():.4e}')

# 保存损失值为 MAT 文件
sio.savemat('loss_history.mat', {'loss': loss_history})

# ##############################
# 结果可视化
# ##############################
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(v_true.detach().cpu().numpy(), cmap='viridis')
plt.title('True Velocity')
plt.subplot(132)
plt.imshow(v_init.detach().cpu().numpy(), cmap='viridis')
plt.title('Initial Velocity Model')
plt.subplot(133)
plt.imshow(v_inv.detach().cpu().numpy(), cmap='viridis')
plt.title('Predicted Velocity Model')
plt.tight_layout()
plt.savefig('fwi_result.png', dpi=300)
