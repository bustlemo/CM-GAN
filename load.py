import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def compute_adjacency_matrix(coords, sigma=100.0):
    """
    根据 GPS 坐标计算邻接矩阵
    :param coords: (N, 2) 形状的 numpy 数组，包含纬度和经度
    :param sigma: 高斯核的带宽参数，PV数据集设为100
    :return: (N, N) 形状的邻接矩阵
    """
    N = coords.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist = np.linalg.norm(coords[i] - coords[j])  # 欧几里得距离
            A[i, j] = np.exp(-dist / sigma)
    return A


class SolarDataset(Dataset):
    def __init__(self, data, adj_matrix, sequence_length=288):
        """
        :param data: (N, 69) 形状的 numpy 数组（归一化后）
        :param sequence_length: 取多少个时间步作为输入（默认 288，表示 24 小时，5分钟粒度）
        """
        self.sequence_length = sequence_length
        self.adj_matrix = adj_matrix
        self.samples = []

        for i in range(0, len(data), sequence_length):
            # 确保最后一个分割段也是完整的 sequence_length
            if i + sequence_length <= len(data):
                self.samples.append(data[i:i + sequence_length].T)  # 转置为(69, sequence_length)

        self.samples = np.array(self.samples)  # (num_samples, 69, sequence_length)

    def __len__(self):
        return len(self.samples)  # 返回num_samples

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.float32)  # 取出第 idx个样本，形状(69, sequence_length)
        return sample, self.adj_matrix  # 返回样本和邻接矩阵


def collate_fn(batch):
    # batch 是一个列表，包含 (sample, adj_matrix) 元组
    power_data = torch.stack([item[0] for item in batch])  # 堆叠功率数据
    adj_matrix = batch[0][1]  # 取第一个样本的 adj_matrix（所有样本共享）
    return power_data, adj_matrix


def load_solar(batch_size=32, sequence_length=288, test_size=0.2, sigma=100.0):
    # 读取 CSV 数据
    df = pd.read_excel('/root/autodl-tmp/CM-GAN/solar.xlsx', header=None)
    data = df.values.astype(np.float32)  # (8640, 69)
    # print("Data shape:", df.shape)  # 打印数据形状
    # print(df.head())  # 打印前几行

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)  # (8640, 69)

    # 读取 GPS 数据并计算邻接矩阵
    gps_df = pd.read_csv('gps.csv')
    gps_coords = gps_df[['latitude', 'longitude']].values  # (69, 2)
    adj_matrix = compute_adjacency_matrix(gps_coords, sigma=sigma)  # 形状 (69, 69)
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)  # 转换为张量

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    # 创建数据集
    train_dataset = SolarDataset(train_data, adj_matrix, sequence_length)
    test_dataset = SolarDataset(test_data, adj_matrix, sequence_length)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader, scaler, adj_matrix