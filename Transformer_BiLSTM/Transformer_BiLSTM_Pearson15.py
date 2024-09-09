import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from VSTLF.data_preprocess.data_pearson15 import train_loader, test_loader, X_train, X_test, data
import numpy as np
# Parameters
input_dim = X_train.shape[2]  # Dimension of input features
n_seq = 7  # Length of input sequence
batch_size = 32  # Batch size for training
output_dim = 1  # Dimension of output features (typically 1 for regression tasks)
hidden_dim = 512  # Dimension of hidden layers in LSTM
embed_dim = 512  # Dimension of embeddings
n_epochs = 200  # Number of training epochs
num_layers = 1  # Number of layers in LSTM and Transformer
learning_rate = 1e-4  # Learning rate for optimizer
is_bidirectional = True  # Whether the LSTM is bidirectional
dropout_prob = 0  # Dropout probability
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPU

## SHAP && person
# Set the factor for bidirectional LSTM
if is_bidirectional:
    D = 2
else:
    D = 1

# Define the DataEmbedding class
class DataEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(DataEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)  # Linear transformation
        self.activation = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Define the Transformer and LSTM hybrid pth
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = DataEmbedding(input_dim, embed_dim)  # Data embedding layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=dropout_prob)  # Transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # Stack of Transformer encoder layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True,
            bidirectional=is_bidirectional
        )  # LSTM layer
        self.decoder = nn.Linear(hidden_dim * D, 1)  # Final linear layer to produce output
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # Create a square subsequent mask for the Transformer
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.embedding(src)  # Apply embedding
        mask = self._generate_square_subsequent_mask(src.size(1)).to(device)  # Generate mask for the Transformer
        src = src.permute(1, 0, 2)  # Permute dimensions for Transformer (sequence, batch, features)
        x = self.transformer_encoder(src, mask)  # Apply Transformer encoder
        x = x.permute(1, 0, 2)  # Permute dimensions back
        hidden_0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)  # Initialize hidden state
        c_0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)  # Initialize cell state
        x, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))  # Apply LSTM
        output = self.decoder(x[:, -1, :])  # Decode the output
        return output

# Initialize pth, loss function, and optimizer
TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(), lr=learning_rate)  # Adam optimizer
criterion = torch.nn.MSELoss(reduction="mean")  # Mean Squared Error loss function

# Training the pth
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

# Save pth parameters
# save_path = 'pth.pth'
# torch.save(TfModel.state_dict(), save_path)

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

# Plot loss curve
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_transformer_lstm.svg", format='svg')
plt.show()

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
