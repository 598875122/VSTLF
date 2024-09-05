from nat_demand.data_handle.data_new import train_loader, test_loader, X_train, timestamps_test
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Parameters
input_dim = X_train.shape[2]  # Dimension of input features
n_seq = 7  # Length of input sequence
batch_size = 32  # Batch size
output_dim = 1  # Dimension of output features (typically 1 for regression tasks)
hidden_dim = 256  # Dimension of LSTM hidden layer
embed_dim = 256  # Dimension of embedding layer
n_epochs = 200  # Number of training epochs
num_layers = 1  # Number of layers in LSTM and Transformer
learning_rate = 1e-3  # Learning rate
is_bidirectional = False  # Whether to use bidirectional LSTM
dropout_prob = 0  # Dropout probability
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU

# Define DataEmbedding class
class DataEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(DataEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)  # Linear transformation layer
        self.activation = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Define the Transformer and LSTM hybrid pth
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = DataEmbedding(input_dim, embed_dim)  # Embedding layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=dropout_prob)  # Transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # Transformer encoder

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True,
            bidirectional=is_bidirectional
        )  # LSTM layer

        self.decoder = nn.Linear(hidden_dim, 1)  # Output layer
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # Create subsequent mask for Transformer
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.embedding(src)  # Embedding layer
        mask = self._generate_square_subsequent_mask(src.size(1)).to(device)  # Generate mask

        src = src.permute(1, 0, 2)  # Transformer expects input in (S, N, E) format, where S is sequence length, N is batch size, E is number of features
        x = self.transformer_encoder(src, mask)  # Pass through Transformer encoder
        x = x.permute(1, 0, 2)  # Convert back to (N, S, E) format for LSTM

        hidden_0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)  # Initialize LSTM hidden state
        c_0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)  # Initialize LSTM cell state
        x, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))  # Pass through LSTM layer
        output = self.decoder(x[:, -1, :])  # Decode output (take the last time step of the sequence)
        return output

# Initialize pth, loss function, and optimizer
TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(), lr=learning_rate)  # Adam optimizer
criterion = torch.nn.MSELoss(reduction="mean")  # Mean Squared Error loss function

# Train the pth
train_losses = []
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

# Set font
plt.rcParams['font.family'] = 'Times New Roman'
# Plot loss curve
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_transformer_lstm.svg", format='svg')
plt.show()

# Model prediction and evaluation
TfModel.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = TfModel(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# Compute evaluation metrics
y_pred = np.array(y_pred)
y_true = np.array(y_true)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
r2 = r2_score(y_true, y_pred)  # R-squared score

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

# Plot true values and predictions
plt.figure(figsize=(15, 5))
plt.plot(timestamps_test, y_true, label='True Values', color='red')
plt.plot(timestamps_test, y_pred, label='Predictions', color='blue')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.ylabel('Actual')
plt.legend()
plt.savefig("prediction_transformer_lstm.svg", format='svg')
plt.show()
