import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SpatialAttention(nn.Module):
    """空间自注意力层"""

    def __init__(self, input_dim, seq_len, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.seq_len = seq_len

        # QS = XS W S q, KS = XSWS k, V S = XSWS v
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # WSq
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # WSk
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # WSv

        # 前馈网络层 - 封装在SpatialAttention类中
        self.W_0 = nn.Linear(hidden_dim, hidden_dim)  # W^S_0
        self.W_1 = nn.Linear(hidden_dim, hidden_dim)  # W^S_1
        self.W_2 = nn.Linear(hidden_dim, hidden_dim)  # W^S_2

        # Softmax层 - 在input_dim维度上应用softmax
        self.softmax = nn.Softmax(dim=-2)

        # 定义邻接矩阵A，可学习的参数
        self.A = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        nn.init.xavier_uniform_(self.A)

        # 定义W_a_S参数矩阵，维度为[input_dim, seq_len, hidden_dim]
        self.W_a_S = nn.Parameter(torch.FloatTensor(input_dim, seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.W_a_S)

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hl=None, adj_matrix=None):
        """
        输入:
            x - [batch_size, input_dim, seq_len, hidden_dim]
            hl - 上一层输出 [batch_size, input_dim, seq_len, hidden_dim]
        输出:
            out - [batch_size, input_dim, seq_len, hidden_dim]
            attention - [batch_size, input_dim, input_dim, hidden_dim]
        """
        if adj_matrix is not None and x.device != adj_matrix.device:
            adj_matrix = adj_matrix.to(x.device)

        batch_size, input_dim, seq_len, _ = x.size()

        # 计算ES = AW_a_S
        # A: [input_dim, input_dim]
        # W_a_S: [input_dim, seq_len, hidden_dim]
        # 结果: [input_dim, seq_len, hidden_dim]
        if adj_matrix is not None:
            # 创建W_a_S参数矩阵，维度为[input_dim, seq_len, hidden_dim]
            # 计算ES = AW_a_S
            ES = torch.einsum('ij,jsh->ish', adj_matrix, self.W_a_S)
            # 扩展ES为[batch_size, input_dim, seq_len, hidden_dim]，使用广播
            ES = ES.unsqueeze(0).expand(batch_size, -1, -1, -1)
            # 计算XS = hl + ES
            if hl is None:
                XS = x + ES
            else:
                XS = hl + ES
        else:
            # 如果没有提供邻接矩阵，直接使用输入
            if hl is None:
                XS = x
            else:
                XS = hl

        # 将XS重塑为[batch_size*input_dim*seq_len, hidden_dim]以应用线性层
        XS_reshaped = XS.reshape(batch_size * input_dim * seq_len, -1)

        # 计算QS, KS, VS
        q = self.W_q(XS_reshaped).view(batch_size, input_dim, seq_len,
                                       -1)  # [batch_size, input_dim, seq_len, hidden_dim]
        k = self.W_k(XS_reshaped).view(batch_size, input_dim, seq_len,
                                       -1)  # [batch_size, input_dim, seq_len, hidden_dim]
        v = self.W_v(XS_reshaped).view(batch_size, input_dim, seq_len,
                                       -1)  # [batch_size, input_dim, seq_len, hidden_dim]

        # 计算注意力分数，使用einsum进行四维张量乘法
        # 在input_dim维度上计算注意力
        # q: [batch_size, input_dim, seq_len, hidden_dim]
        # k: [batch_size, input_dim, seq_len, hidden_dim]
        # 结果: [batch_size, input_dim, input_dim, hidden_dim]
        energy = torch.einsum('bish,bjsh->bijh', q, k) / math.sqrt(self.hidden_dim)

        # 应用softmax得到注意力矩阵 M^S
        attention = self.softmax(energy)  # [batch_size, input_dim, input_dim, hidden_dim]

        # 计算上下文向量，使用einsum进行四维张量乘法
        # attention: [batch_size, input_dim, input_dim, hidden_dim]
        # v: [batch_size, input_dim, seq_len, hidden_dim]
        # 结果: [batch_size, input_dim, seq_len, hidden_dim]
        context = torch.einsum('bijh,bjsh->bish', attention, v)

        # 前馈网络 - Y^S = W^S_1(ReLU(W^S_0(context))))
        # 将context重塑为[batch_size*input_dim*seq_len, hidden_dim]以应用线性层
        context_reshaped = context.reshape(batch_size * input_dim * seq_len, -1)

        ff_output = self.W_0(context_reshaped)
        ff_output = F.relu(ff_output)
        ff_output = self.W_1(ff_output)
        ff_output = F.relu(ff_output)
        ff_output = self.W_2(ff_output)

        # 将输出重塑回[batch_size, input_dim, seq_len, hidden_dim]
        ff_output = ff_output.view(batch_size, input_dim, seq_len, -1)

        # 残差连接和层归一化
        out = self.norm(ff_output + hl if hl is not None else ff_output)

        return out, attention


class TemporalAttention(nn.Module):
    """时间自注意力层"""

    def __init__(self, input_dim, seq_len, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.seq_len = seq_len

        # QT = XT W T q, KT = XTW T k, V T = XTW T v
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # WTq
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # WTk
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # WTv

        # 前馈网络层
        self.W_0 = nn.Linear(hidden_dim, hidden_dim)  # W^T_0
        self.W_1 = nn.Linear(hidden_dim, hidden_dim)  # W^T_1
        self.W_2 = nn.Linear(hidden_dim, hidden_dim)  # W^T_2

        # Softmax层 - 在seq_len维度上应用softmax
        self.softmax = nn.Softmax(dim=-2)

        # 定义时间位置编码，可学习的参数
        self.pos_encoding = nn.Parameter(torch.FloatTensor(seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.pos_encoding)

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hl=None):
        """
        输入:
            x - [batch_size, input_dim, seq_len, hidden_dim]
            hl - 上一层输出 [batch_size, input_dim, seq_len, hidden_dim]，如果为None则初始化
        输出:
            out - [batch_size, input_dim, seq_len, hidden_dim]
            attention - [batch_size, input_dim, input_dim, hidden_dim]
        """
        batch_size, input_dim, seq_len, _ = x.size()

        # 计算ET = 时间位置编码  pos_encoding: [seq_len, hidden_dim]
        # 扩展为[batch_size, input_dim, seq_len, hidden_dim]，使用广播
        ET = self.pos_encoding.unsqueeze(0).unsqueeze(0).expand(batch_size, input_dim, -1, -1)

        # 计算XT = hl + ET
        if hl is None:
            XT = x + ET
        else:
            XT = hl + ET

        # 将XT重塑为[batch_size*input_dim*seq_len, hidden_dim]以应用线性层
        XT_reshaped = XT.reshape(batch_size * input_dim * seq_len, -1)

        # 计算QT, KT, VT
        q = self.W_q(XT_reshaped).view(batch_size, input_dim, seq_len,
                                       -1)  # [batch_size, input_dim, seq_len, hidden_dim]
        k = self.W_k(XT_reshaped).view(batch_size, input_dim, seq_len,
                                       -1)  # [batch_size, input_dim, seq_len, hidden_dim]
        v = self.W_v(XT_reshaped).view(batch_size, input_dim, seq_len,
                                       -1)  # [batch_size, input_dim, seq_len, hidden_dim]

        # 计算注意力分数，使用einsum进行四维张量乘法
        # 在seq_len维度上计算注意力
        # 结果: [batch_size, input_dim, input_dim, hidden_dim]
        energy = torch.einsum('bish,bjsh->bijh', q, k) / math.sqrt(self.hidden_dim)

        # 应用softmax得到注意力矩阵 M^T
        attention = self.softmax(energy)  # [batch_size, input_dim, input_dim, hidden_dim]

        # 计算上下文向量，使用einsum进行四维张量乘法
        # attention: [batch_size, input_dim, input_dim, hidden_dim]
        # v: [batch_size, input_dim, seq_len, hidden_dim]
        # 结果: [batch_size, input_dim, seq_len, hidden_dim]
        context = torch.einsum('bijh,bjsh->bish', attention, v)

        # 前馈网络 - Y^T = W^T_1(ReLU(W^T_0(context))))
        # 将context重塑为[batch_size*input_dim*seq_len, hidden_dim]以应用线性层
        context_reshaped = context.reshape(batch_size * input_dim * seq_len, -1)

        ff_output = self.W_0(context_reshaped)
        ff_output = F.relu(ff_output)
        ff_output = self.W_1(ff_output)
        ff_output = F.relu(ff_output)
        ff_output = self.W_2(ff_output)

        # 将输出重塑回[batch_size, input_dim, seq_len, hidden_dim]
        ff_output = ff_output.view(batch_size, input_dim, seq_len, -1)

        # 残差连接和层归一化
        out = self.norm(ff_output + hl if hl is not None else ff_output)

        return out, attention


class Generator(nn.Module):
    """CM-GAN生成器"""

    def __init__(self, input_dim, seq_len, hidden_dim, num_layers, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1x1 卷积层，将 [batch_size, 1, input_dim, seq_len] 转换为 [batch_size, hidden_dim, input_dim, seq_len]
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1)

        # 初始映射层
        self.initial_mapping = nn.Linear(hidden_dim, hidden_dim)

        # 空间自注意力层
        self.spatial_transformers = nn.ModuleList([
            SpatialAttention(input_dim, seq_len, hidden_dim) for _ in range(num_layers)
        ])

        # 时间自注意力层
        self.temporal_transformers = nn.ModuleList([
            TemporalAttention(input_dim, seq_len, hidden_dim) for _ in range(num_layers)
        ])

        # 合并层 - 将空间和时间特征合并
        self.merge_layers = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)  #[batch_size, input_dim, seq_len, output_dim], output_dim=1 表示功率值

    def forward(self, x, adj_matrix=None):
        # x: 输入数据 [batch_size, input_dim, seq_len]
        batch_size = x.size(0)
        device = x.device

        # 生成噪声 Z [batch_size, seq_len]
        Z = torch.randn(batch_size, self.seq_len, device=device).cuda()
        # 将 Z 扩展到 [batch_size, input_dim, seq_len]
        Z = Z.unsqueeze(1).expand(-1, self.input_dim, -1)
        # 将 Z 添加到 x
        x_with_noise = x + Z  # [batch_size, input_dim, seq_len]
        # 将 x_with_noise 转换为 [batch_size, 1, input_dim, seq_len] 以便于卷积
        x_with_noise = x_with_noise.unsqueeze(1)

        # 通过 1x1 卷积层
        x_conv = self.conv1x1(x_with_noise)  # [batch_size, hidden_dim, input_dim, seq_len]
        # 调整形状为 [batch_size, input_dim, seq_len, hidden_dim]
        x_conv = x_conv.permute(0, 2, 3, 1)  # [batch_size, input_dim, seq_len, hidden_dim]
        # 初始化 hl 为 x_conv
        hl = x_conv

        attention_weights = []

        # Transformer层
        for i in range(self.num_layers):
            # 空间自注意力层
            YS, attn_S = self.spatial_transformers[i](x_conv, hl, adj_matrix)
            # 时间自注意力层
            YT, attn_T = self.temporal_transformers[i](x_conv, hl)
            attention_weights.append((attn_S, attn_T))

            # 拼接 [batch_size, input_dim, seq_len, 3*hidden_dim]
            merged = torch.cat([hl, YS, YT], dim=-1)
            # 将merged重塑为[batch_size*input_dim*seq_len, 3*hidden_dim]以应用线性层
            merged_reshaped = merged.view(batch_size * self.input_dim * self.seq_len, -1)

            # 应用合并层 Wm，得到h_l+1
            hl = self.merge_layers[i](merged_reshaped)
            hl = F.relu(hl)
            # 将hl重塑回[batch_size, input_dim, seq_len, hidden_dim]
            hl = hl.view(batch_size, self.input_dim, self.seq_len, -1)

        # 将final_features重塑为[batch_size*input_dim*seq_len, hidden_dim]以应用线性层
        final_features_reshaped = hl.view(batch_size * self.input_dim * self.seq_len, -1)
        # 输出层
        output = self.output_layer(final_features_reshaped)
        # 将输出重塑回[batch_size, input_dim, seq_len, output_dim]
        output = output.view(batch_size, self.input_dim, self.seq_len, -1)

        return output, attention_weights


class Discriminator(nn.Module):
    """CM-GAN判别器"""
    def __init__(self, input_dim, seq_len, hidden_dim, num_layers, output_dim=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1x1 卷积层，将 [batch_size, 1, input_dim, seq_len] 转换为 [batch_size, hidden_dim, input_dim, seq_len]
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1)

        # 空间自注意力层
        self.spatial_transformers = nn.ModuleList([
            SpatialAttention(input_dim, seq_len, hidden_dim) for _ in range(num_layers)
        ])

        # 时间自注意力层
        self.temporal_transformers = nn.ModuleList([
            TemporalAttention(input_dim, seq_len, hidden_dim) for _ in range(num_layers)
        ])

        # 合并层 - 将空间和时间特征合并
        self.merge_layers = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 输出层
        self.W_d0 = nn.Linear(hidden_dim, hidden_dim)  # W_d0
        self.flatten = nn.Flatten()  # 展平操作
        self.W_d1 = nn.Linear(hidden_dim * input_dim * seq_len, output_dim)  # W_d1
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活

    def forward(self, x, adj_matrix=None):
        """
        x: 输入数据 [batch_size, input_dim, seq_len]
        """
        batch_size = x.size(0)

        x = x.unsqueeze(1)  # 将 x 转换为 [batch_size, 1, input_dim, seq_len] 以便于卷积
        x_conv = self.conv1x1(x)  # [batch_size, hidden_dim, input_dim, seq_len]
        x_conv = x_conv.permute(0, 2, 3, 1)  # [batch_size, input_dim, seq_len, hidden_dim]
        # 初始化 hl
        hl = x_conv

        # Transformer层
        for i in range(self.num_layers):
            # 空间自注意力层
            YS, attn_S = self.spatial_transformers[i](x_conv, hl, adj_matrix)
            # 时间自注意力层
            YT, attn_T = self.temporal_transformers[i](x_conv, hl)

            # 拼接hl, YS, YT [batch_size, input_dim, seq_len, 3*hidden_dim]
            merged = torch.cat([hl, YS, YT], dim=-1)
            # 将merged重塑为[batch_size*input_dim*seq_len, 3*hidden_dim]以应用线性层
            merged_reshaped = merged.view(batch_size * self.input_dim * self.seq_len, -1)

            # 应用合并层 Wm，得到h_l+1
            hl = self.merge_layers[i](merged_reshaped)
            hl = F.relu(hl)
            # 将hl重塑回[batch_size, input_dim, seq_len, hidden_dim]
            hl = hl.view(batch_size, self.input_dim, self.seq_len, -1)

        h_last = hl  # [batch_size, input_dim, seq_len, hidden_dim]
        h_last_reshaped = h_last.reshape(batch_size * self.input_dim * self.seq_len, -1)
        h_after_W_d0 = self.W_d0(h_last_reshaped)  # [batch_size*input_dim*seq_len, hidden_dim]
        h_W_d0_reshaped = h_after_W_d0.view(batch_size, self.input_dim, self.seq_len, -1)
        h_flattened = self.flatten(h_W_d0_reshaped)  # [batch_size, 69 * 288 * 32]
        output = self.W_d1(h_flattened)  # [batch_size, output_dim]
        validity = self.sigmoid(output)  # [batch_size, output_dim]

        return validity
