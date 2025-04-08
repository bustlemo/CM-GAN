import argparse
import os
import torch
import numpy as np
import random
from train import CMGANTrainer


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='CM-GAN for Renewable Energy Scenario Generation')

    # 模型参数
    parser.add_argument('--model', type=str, default='cm_gan', help='模型名称')
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', help='对抗损失类型')
    parser.add_argument('--input_dim', type=int, default=69, help='输入维度（电站数量）')
    parser.add_argument('--hidden_dim', type=int, default=32, help='隐藏层维度 (C=32)')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--sequence_length', type=int, default=288, help='序列长度（5分钟粒度，一天288个时间步）')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--total_step', type=int, default=100, help='总训练步数 (nepoch=1000)')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='生成器学习率 (α=0.0001)')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='判别器学习率 (α=0.0001)')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam优化器beta1参数')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam优化器beta2参数')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='梯度惩罚系数 (λ=10)')
    parser.add_argument('--mu', type=float, default=10.0, help='另一个惩罚系数 (μ=10)')
    parser.add_argument('--ndisc', type=int, default=3, help='每轮判别器迭代次数 (PV数据集为3)')
    parser.add_argument('--ngen', type=int, default=2, help='每轮生成器迭代次数 (PV数据集为2)')

    # 路径参数
    parser.add_argument('--model_save_path', type=str, default='models', help='模型保存路径')
    parser.add_argument('--sample_path', type=str, default='samples', help='样本保存路径')
    parser.add_argument('--log_step', type=int, default=1, help='日志记录间隔')
    parser.add_argument('--sample_step', type=int, default=1, help='样本生成间隔')
    parser.add_argument('--model_save_step', type=int, default=1, help='模型保存间隔')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.sample_path, exist_ok=True)

    # 打印配置信息
    print("=" * 50)
    print("CM-GAN 配置:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)

    # 初始化训练器并开始训练
    trainer = CMGANTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()