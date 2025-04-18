import numpy as np
import matplotlib.pyplot as plt
import torch
import csv


def save_losses_as_csv(g_losses, d_losses, save_path):
    """将损失值保存为CSV格式"""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Generator Loss', 'Discriminator Loss'])

        for i, (g_loss, d_loss) in enumerate(zip(g_losses, d_losses)):
            writer.writerow([i + 1, g_loss, d_loss])
    print(f"损失值已保存到: {save_path}")

def save_model_weights_directly(model, save_path):
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
    epochs = range(1, len(g_losses) + 1)  # 正确设置x轴为epoch数

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
    plt.figure(figsize=(12, 6))

    # 获取数据
    real = real_data[0, station_idx, :].cpu().detach().numpy()
    # 创建完整的时间轴（288个点，对应24小时，每5分钟一个点）
    time_points = range(len(real))
    # 绘制真实数据
    plt.plot(time_points, real, 'r-', linewidth=2, label='Real scenarios')
    # 绘制所有生成数据
    for i in range(generated_data.size(0)):
        gen_curve = generated_data[i, station_idx, :].cpu().detach().numpy()
        if i == 0:
            plt.plot(time_points, gen_curve, 'black', alpha=0.7, label='Generated scenarios')
        else:
            plt.plot(time_points, gen_curve, 'black', alpha=0.7)

    # 设置图表属性
    plt.title('PV data on cGAN')
    plt.xlabel('Time (5 min)')
    plt.ylabel('PV power output (MW)')
    plt.grid(True)
    plt.legend()

    # 设置x轴刻度，每小时显示一个刻度（每12个点）
    hour_ticks = range(0, len(real), 12)
    hour_labels = [f"{h}" for h in range(0, 24)]
    if len(hour_ticks) > len(hour_labels):
        hour_ticks = hour_ticks[:len(hour_labels)]
    elif len(hour_ticks) < len(hour_labels):
        hour_labels = hour_labels[:len(hour_ticks)]
    plt.xticks(hour_ticks, hour_labels)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pv_distribution(real_data, generated_data, station_idx=0, save_path=None):
    """
    绘制真实和生成的光伏输出分布对比图

    参数:
    - real_data: 真实数据，形状为 [batch_size, input_dim, seq_len]
    - generated_data: 生成数据，形状为 [batch_size, input_dim, seq_len] 或 [batch_size, input_dim, seq_len, 1]
    - station_idx: 要绘制的电站索引
    - save_path: 保存路径，如果为None则显示图
    """
    plt.figure(figsize=(6, 5))  # 调整为单独图的大小
    # 处理生成数据的维度，如果是4维则压缩最后一维
    if len(generated_data.shape) == 4:
        generated_data = generated_data.squeeze(-1)

    # 提取所有批次的指定电站数据
    real_samples = real_data[:, station_idx, :].reshape(-1).cpu().numpy()
    gen_samples = generated_data[:, station_idx, :].detach().cpu().numpy().reshape(-1)

    # 绘制分布
    plt.hist(real_samples, bins=50, alpha=0.5, color='blue', label='Real scenarios', density=True)
    plt.hist(gen_samples, bins=50, alpha=0.5, color='red', label='Generated scenarios', density=True)

    plt.xlabel('PV power output (MW)')
    plt.ylabel('Probability')
    plt.title('PV data on cGAN')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_map(attention_weights, save_path=None):
    """
    绘制注意力权重热力图

    参数:
    - attention_weights: 注意力权重，形状为 [seq_len, seq_len]
    - save_path: 保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(10, 8))

    # 绘制热力图
    plt.imshow(attention_weights.cpu().numpy(), cmap='viridis')
    plt.colorbar(label='注意力权重')
    plt.xlabel('序列位置')
    plt.ylabel('序列位置')
    plt.title('自注意力权重图')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_mmd(x, y, gamma=1.0):
    """
    计算最大平均差异(Maximum Mean Discrepancy, MMD)
    参数:
    - x: 第一个数据集，形状为 [batch_size, input_dim, seq_len]
    - y: 第二个数据集，形状为 [batch_size, input_dim, seq_len]
    - gamma: RBF核函数的参数，默认为1.0
    返回:
    - mmd: 最大平均差异值
    """
    if len(x.shape) == 4:
        x = x.squeeze(-1)
    if len(y.shape) == 4:
        y = y.squeeze(-1)
    # 转换为numpy数组并展平
    x_np = x.cpu().detach().numpy().reshape(x.size(0), -1)
    y_np = y.cpu().detach().numpy().reshape(y.size(0), -1)
    # 计算样本数量
    n_x = x_np.shape[0]
    n_y = y_np.shape[0]

    # 计算核矩阵
    def rbf_kernel(x, y, gamma=1.0):
        # 计算欧氏距离的平方
        dist = np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2)
        # 应用RBF核函数
        return np.exp(-gamma * dist)

    # 计算三个核矩阵
    K_XX = rbf_kernel(x_np, x_np, gamma)
    K_XY = rbf_kernel(x_np, y_np, gamma)
    K_YY = rbf_kernel(y_np, y_np, gamma)

    # 计算MMD
    mmd = (np.sum(K_XX) / (n_x * n_x) +
           np.sum(K_YY) / (n_y * n_y) -
           2 * np.sum(K_XY) / (n_x * n_y))

    return np.sqrt(max(mmd, 0))  # 确保结果非负

def evaluate_model(generated_data, test_data, train_data):
    """
    参数:
    - generated_data: 生成的数据，形状为 [batch_size, input_dim, seq_len] 或 [batch_size, input_dim, seq_len, 1]
    - test_data: 测试数据，形状为 [batch_size, input_dim, seq_len]
    - train_data: 训练数据，形状为 [batch_size, input_dim, seq_len]
    """
    # 处理生成数据的维度，如果是4维则压缩最后一维
    if len(generated_data.shape) == 4:
        generated_data = generated_data.squeeze(-1)

    # 计算MMD(X̂,Xte) - 生成数据与测试数据之间的MMD
    mmd_gen_test = compute_mmd(generated_data, test_data, gamma=1.0)
    # 计算MMD(Xtr,Xte) - 训练数据与测试数据之间的MMD
    mmd_train_test = compute_mmd(train_data, test_data, gamma=1.0)
    # 计算改进的MMD指标: MMD = 1 - (MMD(X̂,Xte) - MMD(Xtr,Xte)) / MMD(Xtr,Xte)
    if mmd_train_test > 0:
        mmd_values = 1 - (mmd_gen_test - mmd_train_test) / mmd_train_test
    else:
        mmd_values = 0

    # 返回评估指标
    metrics = {
        'improved_mmd': mmd_values,
        'mmd_gen_test': mmd_gen_test,
        'mmd_train_test': mmd_train_test
    }

    return metrics


def plot_mmd_curve(mmd_values, save_path=None, title='MMD Comparison Between Different Models'):
    """绘制MMD曲线"""
    plt.figure(figsize=(10, 6))

    # 确保有数据
    if 'cGAN' in mmd_values and len(mmd_values['cGAN']) > 0:
        # 使用实际的epoch数量作为x轴
        epochs = range(1, len(mmd_values['cGAN']) + 1)
        # 绘制MMD曲线
        plt.plot(epochs, mmd_values['cGAN'], 'b-', label='cGAN')

        # 添加参考线 - 使用真实数据的MMD值
        if 'Real scenarios data' in mmd_values and len(mmd_values['Real scenarios data']) > 0:
            plt.plot(epochs, mmd_values['Real scenarios data'][:len(epochs)], 'r-', label='Real scenarios data')

        # 设置图表属性
        plt.title(title)
        plt.xlabel('Iteration epochs')
        plt.ylabel('MMD')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 设置合理的y轴范围，但不强制固定
        all_values = mmd_values['cGAN']
        if 'Real scenarios data' in mmd_values:
            all_values = all_values + mmd_values['Real scenarios data'][:len(epochs)]

        if min(all_values) > 0:
            plt.ylim(min(all_values) * 0.8, max(all_values) * 1.2)

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print("警告：MMD值为空，无法绘制曲线")

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