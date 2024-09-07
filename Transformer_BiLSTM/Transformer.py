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
num_layers = 1  # Number of layers in Transformer and LSTM
learning_rate = 1e-3  # Learning rate
is_bidirectional = False  # Whether to use bidirectional LSTM
dropout_prob = 0.1  # Dropout probability
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU

# Define PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)  # Linear transformation layer
        self.activation = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Define the Transformer pth
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, hidden_dim)  # Positional encoding for input normalization
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout_prob)  # Transformer encoder layer with 8 heads
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # Transformer encoder
        self.decoder = nn.Linear(hidden_dim, 1)  # Linear layer for output (could also use Transformer decoder)
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
        src = self.pos_encoder(src)  # Apply positional encoding
        mask = self._generate_square_subsequent_mask(len(src)).to(device)  # Generate mask for Transformer
        output = self.transformer_encoder(src, mask)  # Pass through Transformer encoder
        output = self.decoder(output[:, -1, :])  # Take the last time step and apply decoder
        return output

# Initialize pth, loss function, and optimizer
TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(), lr=learning_rate)  # Adam optimizer
criterion = torch.nn.MSELoss(reduction="mean")  # Mean Squared Error loss function

# Train the pth
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

    # Validation phase
    TfModel.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = TfModel(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))

    print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

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

# Calculate evaluation metrics
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Set font for plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Set font size

# Plot training and validation loss
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(train_losses, label='Training Loss', color='orange')
plt.plot(val_losses, label='Validation Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=12)  # Set legend font size
plt.savefig("loss_transformer_lstm.svg", format='svg')
plt.show()

# Calculate and print evaluation metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
r2 = r2_score(y_true, y_pred)  # R-squared score

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

# Plot true values and predictions
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(timestamps_test, y_true, label='True Values', color='red')
plt.plot(timestamps_test, y_pred, label='Predictions', color='blue')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.ylabel('Actual')
plt.legend(fontsize=12)  # Set legend font size
plt.savefig("prediction_transformer_lstm.svg", format='svg')
plt.show()
