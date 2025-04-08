import os
import time
import torch
import datetime
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import plot_pv_output, save_mmd_values, plot_mmd_curve, plot_pv_distribution, evaluate_model, plot_loss_curve, save_losses_as_csv, save_model_weights_directly

from models import Generator, Discriminator
from load import load_solar


class CMGANTrainer(object):
    def __init__(self, config):
        # 数据加载
        self.train_loader, self.test_loader, self.scaler, self.adj_matrix = load_solar(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            sigma=100.0  # PV数据集的σ参数设为100
        )

        # 模型参数
        self.model = config.model
        self.adv_loss = config.adv_loss
        self.sequence_length = config.sequence_length
        self.input_dim = config.input_dim  # 输入维度 (69个电站)
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size

        # 训练参数
        self.lambda_gp = config.lambda_gp  # 梯度惩罚系数
        self.mu = config.mu  # 另一个惩罚系数
        self.total_step = config.total_step
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # self.pretrained_model = config.pretrained_model
        self.ndisc = config.ndisc  # 判别器迭代次数
        self.ngen = config.ngen  # 生成器迭代次数

        # 路径设置
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # 确保路径存在
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        # 构建模型
        self.build_model()

        self.mmd_values = {
            'cGAN': [],  # 当前模型的MMD值
            # 'cGAN without transformer': [],  # 如果有的话
            'Real scenarios data': [0.25] * self.total_step  # 真实场景数据的MMD值（常数）
        }

        # 获取一批训练数据和测试数据用于MMD计算
        self.train_data_sample = next(iter(self.train_loader))[0].cuda()
        self.test_data_sample = next(iter(self.test_loader))[0].cuda()

        # # 加载预训练模型
        # if self.pretrained_model:
        #     self.load_pretrained_model()

    def build_model(self):
        # 初始化生成器和判别器
        self.G = Generator(
            input_dim=self.input_dim,
            seq_len=self.sequence_length,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1
        ).cuda()

        self.D = Discriminator(
            input_dim=self.input_dim,
            seq_len=self.sequence_length,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1
        ).cuda()

        # 优化器
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, (self.beta1, self.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, (self.beta1, self.beta2)
        )
        # 打印网络结构
        print(self.G)
        print(self.D)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """计算WGAN-GP的梯度惩罚项"""

        # 随机插值系数
        alpha = torch.rand(real_samples.size(0), 1, 1).cuda()
        alpha = alpha.expand_as(real_samples)

        # 插值样本
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # 判别器输出
        d_interpolates = self.D(interpolates, self.adj_matrix.cuda())

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

    def reset_grad(self):
        """重置梯度"""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    # def load_pretrained_model(self):
    #     """加载预训练模型"""
    #     self.G.load_state_dict(torch.load(
    #         os.path.join(self.model_save_path, f'{self.pretrained_model}_G.pth')
    #     ))
    #     self.D.load_state_dict(torch.load(
    #         os.path.join(self.model_save_path, f'{self.pretrained_model}_D.pth')
    #     ))
    #     print(f'加载预训练模型 (step: {self.pretrained_model})...')

    def save_sample(self, fixed_z, step):
        """保存生成的样本"""
        self.G.eval()

        # 从训练集获取一个批次的真实数据
        real_samples, adj_matrix = next(iter(self.train_loader))
        real_samples = real_samples.cuda()
        adj_matrix = adj_matrix.cuda()
        # 生成假样本
        fake_samples = self.G(real_samples, fixed_z, self.adj_matrix.cuda())

        real_data_reshaped = real_samples.cpu()
        fake_data_reshaped = fake_samples.squeeze(-1).detach().cpu()  # 移除最后一个维度

        # 保存对比图
        save_path = os.path.join(self.sample_path, f'sample_{step}.png')
        plot_pv_output(real_data_reshaped, fake_data_reshaped, station_idx=0, save_path=save_path)
        # 分布图
        dist_save_path = os.path.join(self.sample_path, f'distribution_{step}.png')
        plot_pv_distribution(real_data_reshaped, fake_data_reshaped, station_idx=0, save_path=dist_save_path)

        self.G.train()

    def train(self):
        """训练CM-GAN模型"""
        # 固定噪声用于采样
        fixed_z = torch.randn(self.batch_size, self.sequence_length).cuda()

        # 记录损失
        g_losses = []
        d_losses = []
        # 初始化
        g_loss = torch.tensor(0.0).cuda()
        d_loss = torch.tensor(0.0).cuda()

        # 开始训练
        start_time = time.time()
        for step in tqdm(range(self.total_step), desc="训练进度"):
            epoch_g_losses = []
            epoch_d_losses = []
            # 使用tqdm包装数据加载器
            for i, (real_data, adj_matrix) in enumerate(tqdm(self.train_loader, desc=f"Epoch {step+1}", leave=False)):
                # 将数据移至GPU
                real_data = real_data.cuda()  # [batch_size, seq_len, input_dim]

                # ================== 训练判别器 ================== #
                for _ in range(self.ndisc):  # 判别器迭代ndisc次
                    self.D.train()
                    self.G.train()

                    # 真实数据的损失
                    d_real_validity = self.D(real_data, adj_matrix)

                    # 生成假数据
                    z = torch.randn(real_data.size(0), self.sequence_length).cuda()
                    fake_data = self.G(real_data, z, adj_matrix).squeeze(-1)
                    d_fake_validity = self.D(fake_data.detach(), adj_matrix)

                    # 计算WGAN-GP损失
                    d_loss_real = -torch.mean(d_real_validity)
                    d_loss_fake = torch.mean(d_fake_validity)
                    # 梯度惩罚
                    gradient_penalty = self.compute_gradient_penalty(real_data, fake_data)
                    # 判别器总损失
                    d_loss = d_loss_fake + d_loss_real + self.lambda_gp * gradient_penalty

                    # 更新判别器
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                # ================== 训练生成器 ================== #
                for _ in range(self.ngen):  # 生成器迭代ngen次
                    # 生成假数据
                    z = torch.randn(real_data.size(0), self.sequence_length).cuda()
                    fake_data = self.G(real_data, z, adj_matrix)
                    g_fake_validity = self.D(fake_data, adj_matrix)

                    # 计算生成器损失
                    g_loss = -torch.mean(g_fake_validity)
                    # 更新生成器
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()  # 确保更新

                # 记录损失
                epoch_g_losses.append(g_loss.item())
                epoch_d_losses.append(d_loss.item())

                # 清理CUDA缓存
                if i % 5 == 0:  # 每5个批次清理一次
                    torch.cuda.empty_cache()

            # 计算并记录每个epoch的平均损失
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)

            # 在每个step结束后计算MMD
            with torch.no_grad():
                # 生成随机噪声
                z = torch.randn(self.train_data_sample.size(0), self.sequence_length).cuda()
                # 生成样本
                fake_data = self.G(self.train_data_sample, z, self.adj_matrix.cuda())
                # 计算MMD
                metrics = evaluate_model(fake_data, self.test_data_sample, self.train_data_sample)
                # 保存改进的MMD值
                self.mmd_values['cGAN'].append(metrics['improved_mmd'])

            # 打印训练信息
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"已用时 [{elapsed}], Step [{step + 1}/10], "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                      f"Improved MMD: {metrics['improved_mmd']:.6f}")

            # 保存生成的样本
            if (step + 1) % self.sample_step == 0:
                self.save_sample(fixed_z, step + 1)

            # 保存模型
            if (step + 1) % self.model_save_step == 0:
                # 直接保存为txt格式
                save_model_weights_directly(
                    self.G,
                    os.path.join(self.model_save_path, f'{step + 1}_G.txt')
                )
                save_model_weights_directly(
                    self.D,
                    os.path.join(self.model_save_path, f'{step + 1}_D.txt')
                )

            # if (step + 1) % self.model_save_step == 0:
            #     torch.save(self.G.state_dict(),
            #                os.path.join(self.model_save_path, f'{step + 1}_G.pth'))
            #     torch.save(self.D.state_dict(),
            #                os.path.join(self.model_save_path, f'{step + 1}_D.pth'))

                # # 保存损失曲线
                # plt.figure(figsize=(10, 5))
                # plt.plot(g_losses, label='生成器损失')
                # plt.plot(d_losses, label='判别器损失')
                # plt.legend()
                # plt.savefig(os.path.join(self.sample_path, f'loss_{step + 1}.png'))
                # plt.close()
                # 保存MMD值到文件
                save_mmd_values(self.mmd_values, os.path.join(self.model_save_path, f'mmd_values_step_{step + 1}.txt'))
                # 绘制MMD曲线
                plot_mmd_curve(
                    self.mmd_values,
                    save_path=os.path.join(self.model_save_path, f'mmd_curve_step_{step + 1}.png')
                )

        g_path = os.path.join(self.model_save_path, 'generator_final.pth')
        d_path = os.path.join(self.model_save_path, 'discriminator_final.pth')
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
        print(f'已保存最终模型到 {g_path} 和 {d_path}')

        save_model_weights_directly(self.G, os.path.join(self.model_save_path, 'generator_weights.txt'))
        save_model_weights_directly(self.D, os.path.join(self.model_save_path, 'discriminator_weights.txt'))
        save_losses_as_csv(g_losses, d_losses, os.path.join(self.sample_path, 'losses.csv'))
        plot_loss_curve(g_losses, d_losses, save_path=os.path.join(self.sample_path, 'loss_curve.png'))

        save_mmd_values(self.mmd_values, os.path.join(self.model_save_path, 'mmd_values_final.txt'))
        plot_mmd_curve(
            self.mmd_values,
            save_path=os.path.join(self.model_save_path, 'mmd_curve_final.png')
        )