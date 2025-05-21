import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models import Generator, Discriminator
from load import load_solar
from utils import plot_pv_output, plot_pv_distribution, plot_attention_map, plot_loss_curve, save_losses_as_csv, compute_acf, plot_acf
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description='测试CM-GAN模型')

    # 模型参数
    parser.add_argument('--model_path', type=str, default='models/generator_final.pth', help='模型路径')
    parser.add_argument('--disc_path', type=str, default='models/discriminator_final.pth', help='模型路径')
    parser.add_argument('--input_dim', type=int, default=69, help='输入维度（电站数量）')
    parser.add_argument('--hidden_dim', type=int, default=32, help='隐藏层维度 (C=32)')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--sequence_length', type=int, default=288, help='序列长度（5分钟粒度）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='生成样本数量')
    parser.add_argument('--sigma', type=float, default=100.0, help='PV数据集的σ参数')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='梯度惩罚系数')
    parser.add_argument('--mu', type=float, default=0.1, help='重构损失权重')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    _, test_loader, scaler, adj_matrix = load_solar(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        sigma=args.sigma
    )

    # 初始化模型
    generator = Generator(
        input_dim=args.input_dim,
        seq_len=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1
    ).cuda()

    # 初始化判别器
    discriminator = Discriminator(
        input_dim=args.input_dim,
        seq_len=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1
    ).cuda()

    # 加载模型权重
    generator.load_state_dict(torch.load(args.model_path))
    discriminator.load_state_dict(torch.load(args.disc_path))

    generator.eval()
    discriminator.eval()

    # 评估指标
    g_losses = []
    d_losses = []
    all_real_data = []
    all_fake_data = []

    def compute_gradient_penalty(real_samples, fake_samples):
        """计算WGAN-GP的梯度惩罚项"""
        # 随机插值系数
        alpha = torch.rand(real_samples.size(0), 1, 1).cuda()
        alpha = alpha.expand_as(real_samples)

        # 插值样本
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # 判别器输出
        d_interpolates = discriminator(interpolates, adj_matrix.cuda())

        # 创建全1张量
        fake = torch.ones(d_interpolates.size()).cuda()

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 计算梯度惩罚
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    # 生成并评估样本
    for i, (real_data, _) in enumerate(test_loader):
        if i >= args.num_samples:
            break

        real_data = real_data.cuda()

        # 生成样本
        with torch.no_grad():
            fake_data, attention_weights = generator(real_data, adj_matrix.cuda())

            print(f"fake_data shape: {fake_data.shape}")

            # 处理fake_data，确保维度正确
            if len(fake_data.shape) == 4 and fake_data.shape[3] == 1:
                fake_data_squeezed = fake_data.squeeze(3)
            else:
                fake_data_squeezed = fake_data

            # 计算判别器损失
            d_real_validity = discriminator(real_data, adj_matrix.cuda())
            d_fake_validity = discriminator(fake_data_squeezed, adj_matrix.cuda())

            # 计算WGAN-GP损失
            d_loss_real = -torch.mean(d_real_validity)
            d_loss_fake = torch.mean(d_fake_validity)

            # 计算梯度惩罚 (在测试阶段不需要梯度，所以这里不计算)
            # gradient_penalty = compute_gradient_penalty(real_data, fake_data_squeezed)
            gradient_penalty = torch.tensor(0.0).cuda()  # 测试阶段简化

            # 判别器总损失
            d_loss = d_loss_fake + d_loss_real + args.lambda_gp * gradient_penalty

            # 计算生成器损失
            g_fake_validity = discriminator(fake_data_squeezed, adj_matrix.cuda())
            reconstruction_loss = F.mse_loss(fake_data_squeezed, real_data)
            g_loss = -torch.mean(g_fake_validity) + args.mu * reconstruction_loss

            # 记录损失
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            # 打印当前批次的损失
            print(f"批次 {i} - 生成器损失: {g_loss.item():.4f}, 判别器损失: {d_loss.item():.4f}")

            real_data_np = real_data.cpu().numpy()
            fake_data_np = fake_data.cpu().numpy()
            print(f"real_data_np shape: {real_data_np.shape}")
            print(f"fake_data_np shape: {fake_data_np.shape}")

            # 处理四维的fake_data_np，将其转换为三维
            if len(fake_data_np.shape) == 4 and fake_data_np.shape[3] == 1:
                fake_data_np = fake_data_np.squeeze(3)  # 移除最后一个维度，变成 [32, 69, 288]
                print(f"调整后的fake_data_np shape: {fake_data_np.shape}")

            # 原始形状: [batch_size, input_dim, seq_len]
            batch_size, input_dim, seq_len = real_data_np.shape

            # 重塑为[batch_size*seq_len, input_dim]以适应scaler
            real_data_reshaped = np.transpose(real_data_np, (0, 2, 1)).reshape(-1, input_dim)
            fake_data_reshaped = np.transpose(fake_data_np, (0, 2, 1)).reshape(-1, input_dim)

            # 应用反归一化
            real_data_original = scaler.inverse_transform(real_data_reshaped)
            fake_data_original = scaler.inverse_transform(fake_data_reshaped)

            # 重塑回原始形状 [batch_size, input_dim, seq_len]
            real_data_original = np.transpose(real_data_original.reshape(batch_size, seq_len, input_dim), (0, 2, 1))
            fake_data_original = np.transpose(fake_data_original.reshape(batch_size, seq_len, input_dim), (0, 2, 1))

            # 转回PyTorch张量用于后续处理
            real_data_tensor = torch.tensor(real_data_original, dtype=torch.float32)
            fake_data_tensor = torch.tensor(fake_data_original, dtype=torch.float32)

            # 提取反归一化后的 station_idx=0 数据
            real_station_0_original = real_data_tensor[:, 0, :].numpy()  # [batch_size, seq_len]
            fake_station_0_original = fake_data_tensor[:, 0, :].numpy()  # [batch_size, seq_len]

            # 收集数据用于 ACF 计算
            all_real_data.append(real_station_0_original)
            all_fake_data.append(fake_station_0_original)

        # 绘制并保存结果
        plot_pv_output(
            real_data_tensor,
            fake_data_tensor,
            station_idx=0,  # 第一个电站
            save_path=os.path.join(args.output_dir, f'sample_{i}_output.png')
        )

        plot_pv_distribution(
            real_data_tensor,
            fake_data_tensor,
            station_idx=0,  # 第一个电站
            save_path=os.path.join(args.output_dir, f'sample_{i}_distribution.png')
        )

        # 绘制注意力图
        if attention_weights:
            # attention_weights[0] 是 (attn_S, attn_T)
            attn_S, _ = attention_weights[0]  # 取空间注意力
            # attn_S: [batch_size, input_dim, input_dim, hidden_dim]
            attn_S = attn_S.mean(dim=-1)  # 对 hidden_dim 维度取平均值，变为 [batch_size, input_dim, input_dim]
            plot_attention_map(
                attn_S,  # 传入处理后的注意力矩阵
                save_path=os.path.join(args.output_dir, f'sample_{i}_attention.png')
            )

    # 计算 ACF
    all_real_data = np.concatenate(all_real_data, axis=0)  # [total_samples, seq_len]
    all_fake_data = np.concatenate(all_fake_data, axis=0)  # [total_samples, seq_len]

    # 设置合适的max_lag值，确保不超过数据长度
    max_lag = min(288, all_real_data.shape[1] - 1)

    # 计算ACF
    real_acf = compute_acf(all_real_data, max_lag=max_lag)
    gen_acf = compute_acf(all_fake_data, max_lag=max_lag)

    # 绘制ACF图
    plot_acf(
        real_acf,
        gen_acf,
        max_lag=max_lag,
        save_path=os.path.join(args.output_dir, 'acf_plot.png')
    )

    plot_loss_curve(
        g_losses,
        d_losses,
        save_path=os.path.join(args.output_dir, 'test_loss_curve.png')
    )
    save_losses_as_csv(
        g_losses,
        d_losses,
        save_path=os.path.join(args.output_dir, 'test_losses.csv')
    )

    # 打印评估结果
    print("=" * 50)
    print("CM-GAN 模型测试完成，结果已保存到:", args.output_dir)
    print("平均生成器损失:", np.mean(g_losses))
    print("平均判别器损失:", np.mean(d_losses))
    print("=" * 50)

if __name__ == '__main__':
    main()
