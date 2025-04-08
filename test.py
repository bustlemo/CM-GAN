import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models import Generator
from load import load_solar
from utils import plot_pv_output, plot_pv_distribution, plot_attention_map, evaluate_model


def main():
    parser = argparse.ArgumentParser(description='测试CM-GAN模型')

    # 模型参数
    parser.add_argument('--model_path', type=str, default='models/generator_final.pth', help='模型路径')
    parser.add_argument('--input_dim', type=int, default=69, help='输入维度（电站数量）')
    parser.add_argument('--hidden_dim', type=int, default=32, help='隐藏层维度 (C=32)')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--sequence_length', type=int, default=288, help='序列长度（5分钟粒度）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='生成样本数量')
    parser.add_argument('--sigma', type=float, default=100.0, help='PV数据集的σ参数')

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
        output_dim=args.input_dim
    ).cuda()

    # 加载模型权重
    generator.load_state_dict(torch.load(args.model_path))
    generator.eval()

    # 评估指标
    all_metrics = []

    # 生成并评估样本
    for i, (real_data, _) in enumerate(test_loader):
        if i >= args.num_samples:
            break

        real_data = real_data.cuda()

        # 生成随机噪声
        z = torch.randn(real_data.size(0), args.sequence_length).cuda()

        # 生成样本
        with torch.no_grad():
            fake_data = generator(real_data, z, adj_matrix.cuda())
            attention_weights = None

        # 评估模型
        metrics = evaluate_model(real_data, fake_data)
        all_metrics.append(metrics)

        # 绘制并保存结果
        plot_pv_output(
            real_data,
            fake_data,
            station_idx=0,  # 第一个电站
            save_path=os.path.join(args.output_dir, f'sample_{i}_output.png')
        )

        plot_pv_distribution(
            real_data,
            fake_data,
            station_idx=0,  # 第一个电站
            save_path=os.path.join(args.output_dir, f'sample_{i}_distribution.png')
        )

        # 绘制注意力图
        if attention_weights:
            plot_attention_map(
                attention_weights[0],  # 第一层注意力
                save_path=os.path.join(args.output_dir, f'sample_{i}_attention.png')
            )

    # 计算平均指标
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in all_metrics]),
        'mean_wasserstein': np.mean([m['mean_wasserstein'] for m in all_metrics]),
        'max_wasserstein': np.mean([m['max_wasserstein'] for m in all_metrics]),
        'min_wasserstein': np.mean([m['min_wasserstein'] for m in all_metrics])
    }

    # 打印评估结果
    print("=" * 50)
    print("CM-GAN 模型评估结果:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.6f}")
    print("=" * 50)

    # 保存评估结果
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("CM-GAN 模型评估结果:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")


if __name__ == '__main__':
    main()