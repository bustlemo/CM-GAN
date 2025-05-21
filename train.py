import os
import time
import torch
import datetime
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from utils import save_mmd_values, compute_mmd, plot_mmd_curve, plot_loss_curve, save_losses_as_csv, save_model_weights, compute_scaled_mmd
from models import Generator, Discriminator
from load import load_solar


class CMGANTrainer(object):
    def __init__(self, config):
        # 数据加载
        self.train_loader, self.test_loader, self.scaler, self.adj_matrix = load_solar(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            sigma=100.0,  # PV数据集的σ参数设为100
            drop_last=False,  # 不丢弃最后一个不完整批次
            pad_last=True  # 启用批次补齐功能
        )

        # 收集整个训练数据集
        self.train_data = []
        for batch in self.train_loader:
            data, _ = batch
            self.train_data.append(data)
        self.train_data = torch.cat(self.train_data, dim=0).cuda()

        # 收集整个测试数据集
        self.test_data = []
        for batch in self.test_loader:
            data, _ = batch
            self.test_data.append(data)
        self.test_data = torch.cat(self.test_data, dim=0).cuda()

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
        self.ndisc = config.ndisc  # 判别器迭代次数
        self.ngen = config.ngen  # 生成器迭代次数

        # 路径设置
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # 确保路径存在
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        # 构建模型
        self.build_model()

        self.mmd_values = {
            'cGAN': [],  # 记录 MMD(\hat{X}, X_{te})
            # 'cGAN without transformer': [],  # 如果有的话
            'Real scenarios data': [0.2456] * self.total_step,  # 真实场景数据的MMD值（常数）
            'scaled_mmd': []  # 新增 scaled_mmd 记录
        }

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

    def train(self):
        """训练CM-GAN模型"""
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
                real_data = real_data.cuda()  # [batch_size, input_dim, seq_len]

                # ================== 训练判别器 ================== #
                for _ in range(self.ndisc):  # 判别器迭代ndisc次
                    self.D.train()
                    self.G.train()

                    # 真实数据的损失
                    d_real_validity = self.D(real_data, adj_matrix)

                    # 生成假数据
                    fake_data, attention_weights = self.G(real_data, adj_matrix)
                    fake_data = fake_data.squeeze(-1)
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
                    fake_data, attention_weights = self.G(real_data, adj_matrix)
                    fake_data = fake_data.squeeze(-1)
                    g_fake_validity = self.D(fake_data, adj_matrix)

                    # 计算生成器损失
                    reconstruction_loss = F.mse_loss(fake_data, real_data)
                    g_loss = -torch.mean(g_fake_validity) + self.mu * reconstruction_loss
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
                # 生成样本
                fake_data, attention_weights = self.G(self.train_data, self.adj_matrix.cuda())
                # 计算MMD - 只传入生成的数据部分
                mmd_hat_te = compute_mmd(fake_data, self.test_data, gamma=1.0, input_dim=self.input_dim,
                                         seq_len=self.sequence_length)
                self.mmd_values['cGAN'].append(mmd_hat_te.item())

                # 计算 scaled MMD 用于保存
                scaled_mmd = compute_scaled_mmd(fake_data, self.test_data, self.train_data, gamma=1.0,
                                                input_dim=self.input_dim, seq_len=self.sequence_length)
                self.mmd_values['scaled_mmd'].append(scaled_mmd.item())

            # 打印训练信息
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"已用时 [{elapsed}], Step [{step + 1}/1000], "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                      f"MMD(\hat{{X}}, X_{{te}}): {mmd_hat_te.item():.6f}, Scaled MMD: {scaled_mmd.item():.6f}")

            # 保存模型
            if (step + 1) % self.model_save_step == 0:
                # 保存MMD值到文件
                save_mmd_values({'scaled_mmd': self.mmd_values['scaled_mmd']},
                                os.path.join(self.model_save_path, f'scaled_mmd_step_{step + 1}.txt'))
                # 绘制 MMD(\hat{X}, X_{te}) 曲线
                plot_mmd_curve(
                    self.mmd_values,
                    save_path=os.path.join(self.model_save_path, f'mmd_curve_step_{step + 1}.png')
                )

        g_path = os.path.join(self.model_save_path, 'generator_final.pth')
        d_path = os.path.join(self.model_save_path, 'discriminator_final.pth')
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
        print(f'已保存最终模型到 {g_path} 和 {d_path}')

        save_model_weights(self.G, os.path.join(self.model_save_path, 'generator_weights.txt'))
        save_model_weights(self.D, os.path.join(self.model_save_path, 'discriminator_weights.txt'))
        save_losses_as_csv(g_losses, d_losses, os.path.join(self.sample_path, 'losses.csv'))
        plot_loss_curve(g_losses, d_losses, save_path=os.path.join(self.sample_path, 'loss_curve.png'))

        save_mmd_values({'scaled_mmd': self.mmd_values['scaled_mmd']},
                        os.path.join(self.model_save_path, 'scaled_mmd_final.txt'))
        plot_mmd_curve(
            self.mmd_values,
            save_path=os.path.join(self.model_save_path, 'mmd_curve_final.png')
        )
