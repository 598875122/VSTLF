import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('continuous dataset.csv')

# 确保datetime列为datetime类型
data['datetime'] = pd.to_datetime(data['datetime'])


# 设置datetime列为索引（可选）
data.set_index('datetime', inplace=True)
# data_target = data
# data = data.drop(['nat_demand'], axis=1)
# 用于存储处理后的数据
X = []
y = []

# 滑动窗口大小为7小时
window_size = 7

# 遍历数据
for i in range(len(data) - window_size):
    # 提取窗口内的特征
    window_features = data.iloc[i:i + window_size].values
    # 提取预测目标
    target = data['nat_demand'].iloc[i + window_size]

    # 将特征和目标分别添加到X和y中
    X.append(window_features)
    y.append(target)

# 转换为numpy数组
X = np.array(X)
y = np.array(y)
# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset[0])