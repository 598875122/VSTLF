from nat_demand.data_handle.data_new import train_loader, test_loader, X_train, timestamps_test
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# parameters
input_dim = X_train.shape[2]
n_seq = 7
batch_size = 32
output_dim = 1
hidden_dim = 256
embed_dim = 256
n_epochs = 200
num_layers = 1
learning_rate = 1e-3
# weight_decay = 1e-6
is_bidirectional = False
dropout_prob = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
        # self.fc = nn.Linear(hidden_dim, 32)
        self.decoder = nn.Linear(hidden_dim, 1)
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

        src = src.permute(1, 0,
                          2)  # Transformer expects (S, N, E) where S is sequence length, N is batch size, E is feature number

        x = self.transformer_encoder(src, mask)
        x = x.permute(1, 0, 2)  # Convert back to (N, S, E) for LSTM
        hidden_0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)
        x, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))
        # x = self.fc(x[:, -1, :])
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

    print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}')

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
# 画出loss图
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_transformer_lstm.svg", format='svg')
plt.show()

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

plt.figure(figsize=(15, 5))
plt.plot(timestamps_test, y_true, label='True Values', color='red')
plt.plot(timestamps_test, y_pred, label='Predictions', color='blue')
plt.xticks(rotation=45)  # 设置横坐标标签旋转45度
# plt.title('Comparison of True Values and Predictions')
plt.ylabel('Actual')
plt.legend()
plt.savefig("prediction_transformer_lstm.svg", format='svg')
plt.show()