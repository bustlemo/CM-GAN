import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
from statsmodels.tsa.stattools import acf


def save_losses_as_csv(g_losses, d_losses, save_path):
    """将损失值保存为CSV格式"""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Generator Loss', 'Discriminator Loss'])

        for i, (g_loss, d_loss) in enumerate(zip(g_losses, d_losses)):
            writer.writerow([i + 1, g_loss, d_loss])
    print(f"损失值已保存到: {save_path}")

def save_model_weights(model, save_path):
    """直接将模型权重保存为可读的txt格式"""
    with open(save_path, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"Shape: {param.shape}\n")
            # 保存完整参数值
            param_data = param.data.cpu().numpy().flatten()
            f.write(f"Values: {param_data.tolist()}\n")
            f.write("-" * 50 + "\n")
    print(f"模型权重已保存到: {save_path}")

def plot_loss_curve(g_losses, d_losses, save_path=None):
    """绘制损失函数曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(g_losses) + 1)  # x轴为epoch数

    plt.plot(epochs, g_losses, 'b-', label='Generator Loss')
    plt.plot(epochs, d_losses, 'r-', label='Discriminator Loss')

    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pv_output(real_data, generated_data, station_idx=0, save_path=None):
    """绘制真实和生成的PV输出对比图"""
    plt.figure(figsize=(12, 8))
    # 对真实数据在batch维度上取均值，得到一条曲线
    real_mean = real_data[:, station_idx, :].mean(dim=0).cpu().detach().numpy()
    # 创建完整的时间轴（288个点，对应24小时，每5分钟一个点）
    time_points = range(len(real_mean))

    # 绘制所有生成数据（黑色线条）
    for i in range(generated_data.size(0)):
        gen_curve = generated_data[i, station_idx, :].cpu().detach().numpy()
        if i == 0:
            plt.plot(time_points, gen_curve, 'black', linewidth=0.8, alpha=0.7, label='Generated scenarios')
        else:
            plt.plot(time_points, gen_curve, 'black', linewidth=0.8, alpha=0.7)

    # 绘制真实数据均值（红色线条）
    plt.plot(time_points, real_mean, 'r-', linewidth=2.0, label='Real scenarios')

    # 设置图表属性
    plt.title('PV Power Output Scenarios')
    plt.xlabel('Time (5min)')
    plt.ylabel('PV power output (MW)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    # 设置x轴刻度，每30个点显示一个刻度
    x_ticks = range(0, len(real_mean), 30)
    x_labels = [f"{i}" for i in x_ticks]
    plt.xticks(x_ticks, x_labels)

    # 获取y轴范围并设置刻度
    y_min = 0
    y_max = max(real_mean.max(), generated_data[:, station_idx, :].max().item()) * 1.1
    y_ticks = np.linspace(0, y_max, 7)  # 创建7个均匀分布的刻度点
    plt.yticks(y_ticks)
    plt.ylim(y_min, y_max)

    # 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pv_distribution(real_data, fake_data, station_idx=0, save_path=None):
    """
    绘制真实和生成的PV输出分布
    real_data, fake_data: [batch_size, input_dim, seq_len]
    station_idx: 要绘制的电站索引
    """
    # 创建两个子图，一个用于PDF，一个用于CDF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 提取指定电站的数据
    real_values = real_data[:, station_idx, :].flatten().numpy() if isinstance(real_data, torch.Tensor) else real_data[
                                                                                                             :,
                                                                                                             station_idx,
                                                                                                             :].flatten()
    gen_values = fake_data[:, station_idx, :].flatten().numpy() if isinstance(fake_data, torch.Tensor) else fake_data[:,
                                                                                                            station_idx,
                                                                                                            :].flatten()

    # 设置x轴刻度
    x_ticks1 = np.linspace(0, 100, 6)  # 0, 20, 40, 60, 80, 100
    x_ticks2 = np.linspace(0, 100, 21)

    # 在第一个子图中绘制PDF直方图
    ax1.hist(real_values, bins=25, alpha=0.7, density=True, color='blue', label='Real scenarios')
    hist_values, bin_edges = np.histogram(gen_values, bins=25, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 使用plot绘制生成数据的分布曲线，删除bins和density参数
    ax1.plot(bin_centers, hist_values, color='red', label='Generated scenarios')

    ax1.set_xlabel('PV power output (MW)')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'PDF of PV Power Output for Station {station_idx}')
    ax1.set_xticks(x_ticks1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 在第二个子图中绘制CDF累积图
    hist_real, bin_edges = np.histogram(real_values, bins=25, density=True)
    cum_real = np.cumsum(hist_real) * (bin_edges[1] - bin_edges[0])

    hist_gen, _ = np.histogram(gen_values, bins=25, density=True)
    cum_gen = np.cumsum(hist_gen) * (bin_edges[1] - bin_edges[0])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 使用plot绘制CDF曲线
    ax2.plot(bin_centers, cum_real, color='blue', label='Real scenarios')
    ax2.plot(bin_centers, cum_gen, color='red', label='Generated scenarios')

    ax2.set_xlabel('PV power output (MW)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'CDF of PV Power Output for Station {station_idx}')
    ax2.set_xticks(x_ticks2)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到 {save_path}")

    plt.close()


def plot_attention_map(attention, save_path):
    # attention: [batch_size, input_dim, input_dim]
    attention = attention[0].cpu().numpy()  # 取第一个样本，形状 [input_dim, input_dim]
    plt.imshow(attention, cmap='viridis')
    plt.colorbar()
    plt.title("Spatial Attention Map")
    plt.savefig(save_path)
    plt.close()


def gaussian_kernel(x, y, gamma=1.0):
    """计算高斯核函数"""
    return torch.exp(-gamma * (x - y) ** 2)

def compute_mmd(X, Y, gamma=1.0, input_dim=69, seq_len=288, batch_size=1024):
    """
    计算最大平均差异(Maximum Mean Discrepancy, MMD)
    参数:
    - X: 第一个数据集，形状为 [batch_size, input_dim, seq_len]
    - Y: 第二个数据集，形状为 [batch_size, input_dim, seq_len]
    - gamma: RBF核函数的参数，默认为1.0
    返回:
    - mmd: 最大平均差异值
    """
    if X.dim() == 4:
        X = X.squeeze(-1)
    if Y.dim() == 4:
        Y = Y.squeeze(-1)

    total_samples_X = X.shape[0]
    total_samples_Y = Y.shape[0]

    # 重塑数据
    X_flat = X.view(total_samples_X, input_dim, seq_len).permute(1, 0, 2).reshape(input_dim, total_samples_X * seq_len)
    Y_flat = Y.view(total_samples_Y, input_dim, seq_len).permute(1, 0, 2).reshape(input_dim, total_samples_Y * seq_len)

    # 总点数
    total_points_X = total_samples_X * seq_len
    total_points_Y = total_samples_Y * seq_len

    # 初始化 MMD 的各项
    sum_k_X = torch.zeros(input_dim, device=X.device)
    sum_k_Y = torch.zeros(input_dim, device=Y.device)
    sum_k_XY = torch.zeros(input_dim, device=X.device)

    # 分批计算 k(X, X)
    for i in range(0, total_points_X, batch_size):
        X_i = X_flat[:, i:i + batch_size]
        for j in range(0, total_points_X, batch_size):
            X_j = X_flat[:, j:j + batch_size]
            X_diff = X_i.unsqueeze(2) - X_j.unsqueeze(1)  # [input_dim, batch_i, batch_j]
            K_X = torch.exp(-gamma * X_diff ** 2)
            if i == j:
                # 减去对角线项
                batch_points = X_i.shape[1]
                sum_k_X += (K_X.sum(dim=[1, 2]) - K_X.diagonal(dim1=1, dim2=2).sum(dim=1))
            else:
                sum_k_X += K_X.sum(dim=[1, 2])
        torch.cuda.empty_cache()  # 清理显存

    # 归一化
    sum_k_X = sum_k_X / (total_points_X * (total_points_X - 1))

    # 分批计算 k(Y, Y)
    for i in range(0, total_points_Y, batch_size):
        Y_i = Y_flat[:, i:i + batch_size]
        for j in range(0, total_points_Y, batch_size):
            Y_j = Y_flat[:, j:j + batch_size]
            Y_diff = Y_i.unsqueeze(2) - Y_j.unsqueeze(1)
            K_Y = torch.exp(-gamma * Y_diff ** 2)
            if i == j:
                batch_points = Y_i.shape[1]
                sum_k_Y += (K_Y.sum(dim=[1, 2]) - K_Y.diagonal(dim1=1, dim2=2).sum(dim=1))
            else:
                sum_k_Y += K_Y.sum(dim=[1, 2])
        torch.cuda.empty_cache()

    sum_k_Y = sum_k_Y / (total_points_Y * (total_points_Y - 1))

    # 分批计算 k(X, Y)
    for i in range(0, total_points_X, batch_size):
        X_i = X_flat[:, i:i + batch_size]
        for j in range(0, total_points_Y, batch_size):
            Y_j = Y_flat[:, j:j + batch_size]
            XY_diff = X_i.unsqueeze(2) - Y_j.unsqueeze(1)
            K_XY = torch.exp(-gamma * XY_diff ** 2)
            sum_k_XY += K_XY.sum(dim=[1, 2])
        torch.cuda.empty_cache()

    sum_k_XY = sum_k_XY / (total_points_X * total_points_Y)

    # 合并计算 MMD
    mmd = (sum_k_X + sum_k_Y - 2 * sum_k_XY).mean()
    return mmd

def compute_scaled_mmd(X_hat, X_te, X_tr, gamma=1.0, input_dim=69, seq_len=288):
    """
    计算批次数据的 scaled MMD
    参数:
        X_hat: 生成数据 [batch_size, N, T]
        X_te: 测试数据 [batch_size, N, T]
        X_tr: 训练数据 [batch_size, N, T]
        gamma: 高斯核参数, 默认为1
    返回:
        scaled_mmd: [batch_size]
    """
    mmd_hat_te = compute_mmd(X_hat, X_te, gamma, input_dim, seq_len)
    mmd_tr_te = compute_mmd(X_tr, X_te, gamma, input_dim, seq_len)
    if mmd_tr_te > 0:
        scaled_mmd = 1 - (mmd_hat_te - mmd_tr_te) / mmd_tr_te
    else:
        scaled_mmd = 0
    return scaled_mmd


def plot_mmd_curve(mmd_values, save_path=None, title='MMD Comparison Between Different Models'):
    """绘制MMD曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(mmd_values['cGAN']) + 1)
    plt.plot(epochs, mmd_values['cGAN'], 'b-', label='cGAN')
    if 'Real scenarios data' in mmd_values:
        plt.plot(epochs, mmd_values['Real scenarios data'][:len(epochs)], 'r-', label='Real scenarios data')
    plt.title(title)
    plt.xlabel('Iteration epochs')
    plt.ylabel('MMD')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_mmd_values(mmd_values, file_path):
    """
    保存MMD值到文件
    参数:
    - mmd_values: 字典，包含不同模型的MMD值列表
    - file_path: 保存路径
    """
    with open(file_path, 'w') as f:
        for model_name, values in mmd_values.items():
            f.write(f"{model_name}:\n")
            for i, value in enumerate(values):
                f.write(f"Epoch {i + 1}: {value:.6f}\n")
            f.write("\n")


def compute_acf(data, max_lag=288):
    """
    计算时间序列的自相关函数（ACF）
    :param data: numpy数组，形状为 [seq_len] 或 [num_samples, seq_len]
    :param max_lag: 最大lag值，默认288
    :return: ACF值，形状为 [max_lag + 1]
    """
    if data.ndim == 1:
        max_lag = min(max_lag, len(data) - 1)
        acf_values = acf(data, nlags=max_lag, fft=True)
    elif data.ndim == 2:
        # 对每个样本计算ACF，然后取平均
        max_lag = min(max_lag, data.shape[1] - 1)
        acfs = []
        for i in range(data.shape[0]):
            sample_acf = acf(data[i], nlags=max_lag, fft=True)
            acfs.append(sample_acf)
        acf_values = np.mean(acfs, axis=0)
    else:
        raise ValueError("data must be 1D or 2D array")

        # 确保返回的长度为 max_lag + 1
    expected_length = max_lag + 1
    if len(acf_values) < expected_length:
        # 如果长度不足，填充 0
        acf_values = np.pad(acf_values, (0, expected_length - len(acf_values)), mode='constant', constant_values=0)
    elif len(acf_values) > expected_length:
        # 如果长度过长，截断
        acf_values = acf_values[:expected_length]

    return acf_values

def plot_acf(real_acf, gen_acf, max_lag=288, save_path=None):
    """
    绘制真实数据和生成数据的ACF折线图
    :param real_acf: 真实数据的ACF值，形状为 [max_lag + 1]
    :param gen_acf: 生成数据的ACF值，形状为 [max_lag + 1]
    :param max_lag: 最大lag值，默认288
    :param save_path: 保存路径
    """
    min_length = min(len(real_acf), len(gen_acf), max_lag + 1)
    real_acf = real_acf[:min_length]
    gen_acf = gen_acf[:min_length]

    # 创建对应的lag值数组
    lags = np.arange(min_length)

    plt.figure(figsize=(12, 6))
    plt.plot(lags, real_acf, 'r-', label='Real')
    plt.plot(lags, gen_acf, 'b-', label='Generated')
    plt.title('Autocorrelation Function (ACF) for Station 0')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation Coefficient')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 设置x轴刻度，每5个点标注一次
    x_ticks = np.arange(0, min_length, 10)
    plt.xticks(x_ticks)

    # 设置y轴范围
    plt.ylim(-1, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ACF图已保存到 {save_path}")
        plt.close()
    else:
        plt.show()
