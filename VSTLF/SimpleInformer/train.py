import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from main_Informer import Informer
from Train.utils.dataset_built import train_loader, test_loader

# 模型参数
enc_in = 12  # 输入特征的维度
dec_in = enc_in  # 解码器输入的维度与编码器相同
c_out = 1  # 输出特征的维度（对于回归任务通常是 1）
seq_len = 1  # 序列长度为1，因为我们进行单步预测
label_len = 1  # 标签长度，对于单步预测也是1
out_len = 1  # 输出序列的长度，对于单步预测通常是 1
e_layers = 2  # 编码器层数
d_layers = 2  # 解码器层数
hidden_dim = 256  # 前馈网络隐藏层的维度因子
n_heads = 8  # 注意力头的数量
dropout = 0.1  # Dropout概率
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len, e_layers, d_layers, hidden_dim, n_heads, dropout).to(device)


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

losses = []
# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets, time_f) in enumerate(train_loader):
        # print(inputs.shape,targets.shape,time_f.shape)
        inputs, targets, time_f = inputs.to(device), targets.to(device), time_f.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, inputs, time_f)  # 由于单步预测，输入数据用作编码器和解码器的输入
        targets = targets.unsqueeze(1).unsqueeze(2)
        loss = criterion(outputs, targets)  # 损失函数不需要添加额外的维度
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 绘制损失曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
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
print(y_pred, y_true)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

# 绘制预测值和真实值的折线图进行对比
plt.figure(figsize=(15, 5))
plt.plot(y_true, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='red', linestyle='dashed')
plt.title('Comparison of True Values and Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.savefig("prediction_LstmEn_De.svg", format='svg')
plt.show()
