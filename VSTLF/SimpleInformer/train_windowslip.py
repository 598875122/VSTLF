import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from main_Informer import Informer
from dataProcess import train_loader, test_loader, X_train
from torchinfo import summary
# 模型参数

enc_in = X_train.shape[2]  # 输入特征的维度
print(enc_in)
dec_in = enc_in  # 解码器输入的维度与编码器相同
c_out = 1  # 输出特征的维度（对于回归任务通常是 1）
seq_len = 7  # 序列长度为1，因为我们进行单步预测
label_len = 1  # 标签长度，对于单步预测也是1
out_len = 1  # 输出序列的长度，对于单步预测通常是 1
e_layers = 1  # 编码器层数
d_layers = 1  # 解码器层数
hidden_dim = 512  # 前馈网络隐藏层的维度因子
n_heads = 4  # 注意力头的数量
dropout = 0.1  # Dropout概率
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len, e_layers,
                 d_layers, hidden_dim, n_heads, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

losses = []
# 训练循环
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets, time_f) in enumerate(train_loader):
        # 移动数据到设备
        inputs, targets, time_f = inputs.to(device), targets.to(device), time_f.to(device)
        # print(inputs.shape,targets.shape,time_f.shape)
        optimizer.zero_grad()
        outputs = model(inputs, inputs, time_f)  # 由于单步预测，输入数据用作编码器和解码器的输入

        outputs = outputs.squeeze(-1)  # 去掉最后一个维度，使输出形状变为 [batch_size]
        targets = targets.squeeze(-1)  # 确保目标的形状也是 [batch_size]
        loss = criterion(outputs, targets)  # 损失函数不需要添加额外的维度
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Informer Training Loss Curve')
plt.legend()
plt.savefig("Loss_Informer1.svg", format='svg')
plt.show()

# 将模型切换到评估模式
model.eval()

# 存储真实值和预测值
true_values = []
predicted_values = []

# 不计算梯度以加快计算速度
with torch.no_grad():
    for inputs, labels, time_f in test_loader:
        # 移动数据到设备
        inputs, labels, time_f = inputs.to(device), labels.to(device), time_f.to(device)
        outputs = model(inputs, inputs, time_f)  # 同样，解码器输入简化处理
        true_values.extend(labels.cpu().numpy())  # 确保从 GPU 移动到 CPU 再转换为 numpy
        y_pred = outputs.squeeze().cpu().numpy()  # 确保从 GPU 移动到 CPU 再转换为 numpy
        predicted_values.extend(y_pred)



# 计算评估指标
y_pred = np.array(predicted_values)
y_true = np.array(true_values)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加 DejaVu Sans 以支持负号
# Plot comparison of true values and predictions
plt.figure(figsize=(18, 8))
plt.plot(y_true, label='True Values', color='r', linestyle='-', marker='o', markersize=4)
plt.plot(y_pred, label='Predictions', color='y', linestyle='--', marker='x', markersize=4)
plt.title('Informer Comparison of True Values and Predictions', fontsize=20)
plt.ylabel('充电量(Ah)', fontsize=16)
# 设置坐标轴刻度值的字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(True)
plt.legend()
plt.savefig("prediction_Informer1.svg", format='svg')
plt.show()

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(losses, columns=["Informer"])
# 保存到 CSV 文件
df.to_csv('loss_Informer.csv', index=False)

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(y_pred, columns=["Informer"])

# 保存到 CSV 文件
df.to_csv('pred_Informer.csv', index=False)