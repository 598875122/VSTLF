import shap
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nat_demand.data_handle.data_new import train_loader, test_loader, X_train, X_test, data, timestamps_test

# parameters
input_dim = X_train.shape[2]
n_seq = 7
batch_size = 32
output_dim = 1
hidden_dim = 256
embed_dim = 256
n_epochs = 200
num_layers = 1
learning_rate = 1e-4
is_bidirectional = False
dropout_prob = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if is_bidirectional:
    D = 2
else:
    D = 1


# 定义DataEmbedding类
class DataEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(DataEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


# 定义Transformer和LSTM混合模型
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = DataEmbedding(input_dim, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True,
            bidirectional=is_bidirectional
        )
        self.decoder = nn.Linear(hidden_dim * D, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.embedding(src)
        mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
        src = src.permute(1, 0, 2)
        x = self.transformer_encoder(src, mask)
        x = x.permute(1, 0, 2)
        hidden_0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)
        c_0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)
        x, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))
        output = self.decoder(x[:, -1, :])
        return output


# 初始化模型、损失函数和优化器
TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss(reduction="mean")

# 训练模型
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    TfModel.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = TfModel(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # 验证阶段
    #TfModel.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = TfModel(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))

    print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
# 保存模型参数
save_path = '../model.pth'
torch.save(TfModel.state_dict(), save_path)

# 模型预测和评估
TfModel.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = TfModel(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 设置字体大小

# 绘制图形
plt.figure(figsize=(10, 6))  # 设置图的大小
plt.plot(train_losses, label='Training Loss',color='orange')
plt.plot(val_losses, label='Validation Loss',color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=12)  # 设置图例字体大小
plt.savefig("loss_transformer_lstm.svg", format='svg')
plt.show()

# 计算评估指标
y_pred = np.array(y_pred)
y_true = np.array(y_true)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

# 绘制预测值和真实值的折线图进行对比
plt.figure(figsize=(10, 6))  # 设置图的大小
plt.plot(timestamps_test, y_true, label='True Values', color='red')
plt.plot(timestamps_test, y_pred, label='Predictions', color='blue')
plt.xticks(rotation=45)  # 设置横坐标标签旋转45度
# plt.title('Comparison of True Values and Predictions')
plt.ylabel('Actual')
plt.legend(fontsize=12)  # 设置图例字体大小
plt.savefig("prediction_transformer_lstm.svg", format='svg')
plt.show()

# # SHAP分析
# # SHAP分析
# background = X_train[:100].to(device)
# test_data = X_test[:100].to(device)
#
# # 使用 GradientExplainer 计算 SHAP 值
# TfModel.train()
# explainer = shap.GradientExplainer(TfModel, background)
# TfModel.eval()
# TfModel.train()
# shap_values = explainer.shap_values(test_data)
# TfModel.eval()
# # 获取特征名称
#
# feature_names = data.columns
#
# # 展平 shap_values 以匹配 X_display
# shap_values = np.array(shap_values).reshape(-1, X_train.shape[2])
# X_display = test_data.cpu().numpy().reshape(-1, X_train.shape[2])
#
# # 可视化第一个prediction的解释
# shap.initjs()
# force_plot = shap.force_plot(np.mean(shap_values), shap_values[0], X_display[0], feature_names=feature_names)
#
# # 保存为HTML文件
# shap.save_html("force_plot_0.html", force_plot)
# # 绘制 SHAP 横向柱状图
# shap.summary_plot(shap_values, X_display, feature_names=feature_names, plot_type="bar")
