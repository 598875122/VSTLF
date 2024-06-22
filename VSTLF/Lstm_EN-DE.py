import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_new import train_loader, test_loader, X_train

# parameters
input_dim = X_train.shape[2]
n_seq = 7
batch_size = 32
output_dim = 1
hidden_dim = 256
n_epochs = 200
num_layers = 1
learning_rate = 1e-3
weight_decay = 1e-6
is_bidirectional = False
dropout_prob = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if is_bidirectional:
    D = 2
else:
    D = 1

# 定义LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.bidirectional = is_bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)
        c0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn

# 定义LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.bidirectional = is_bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim * D, self.output_dim)

    def forward(self, x, hn, cn):
        out_lstm, (hn, cn) = self.lstm(x, (hn, cn))
        out_lstm = out_lstm[:, -1, :]
        out = self.fc(out_lstm)
        return out, hn, cn

# 定义Encoder-Decoder模型
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        encoder_hidden, encoder_cell = self.encoder(x)
        decoder_input = x
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        outputs = torch.zeros(x.size(0), x.size(1), self.output_dim).to(device)
        for t in range(x.size(1)):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input[:, t:t+1, :], decoder_hidden, decoder_cell)
            outputs[:, t, :] = decoder_output
        return outputs[:, -1, :]

# 初始化模型、损失函数和优化器
LstmEn_De = EncoderDecoder().to(device)
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(params=LstmEn_De.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 训练模型
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    LstmEn_De.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = LstmEn_De(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}')

# 画出loss图
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_LstmEn_De.svg", format='svg')
plt.show()

# 模型预测和评估
LstmEn_De.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = LstmEn_De(inputs)
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
plt.plot(y_true, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='red', linestyle='dashed')
plt.title('Comparison of True Values and Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.savefig("prediction_LstmEn_De.svg", format='svg')
plt.show()
