import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
data = pd.read_csv('../continuous dataset.csv')
# 确保datetime列为datetime类型
data['datetime'] = pd.to_datetime(data['datetime'])

# 设置datetime列为索引（可选）
data.set_index('datetime', inplace=True)
data_target = data['nat_demand']

# 计算与目标值的 Pearson 相关系数
correlations = data.corrwith(data_target).abs()
top_15_features = correlations.nlargest(5).index

cols = ["nat_demand","W2M_san","T2M_toc","T2M_dav","T2M_san"]
data = data[cols]
# 选择相关性最高的 15 个特征
#data = data[top_15_features]

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
    target = data_target.iloc[i + window_size]

    # 将特征和目标分别添加到X和y中
    X.append(window_features)
    y.append(target)

# 转换为numpy数组
X = np.array(X)
y = np.array(y)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 手动按顺序分割数据集
test_size = 0.1
split_index = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset[0])
