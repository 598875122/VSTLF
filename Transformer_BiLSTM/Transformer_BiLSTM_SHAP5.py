import shap
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from VSTLF.data_preprocess.data_raw16 import train_loader, test_loader, X_train, X_test, data, timestamps_test

# Parameters
input_dim = X_train.shape[2]  # Number of input features
n_seq = 7  # Length of input sequence
batch_size = 32  # Batch size
output_dim = 1  # Dimension of output features (typically 1 for regression tasks)
hidden_dim = 512  # Dimension of LSTM hidden layer
embed_dim = 512  # Dimension of embedding layer
n_epochs = 200  # Number of training epochs
num_layers = 1  # Number of layers in Transformer and LSTM
learning_rate = 1e-4  # Learning rate
is_bidirectional = True  # Whether to use bidirectional LSTM
dropout_prob = 0  # Dropout probability
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU

# Determine the number of directions for LSTM
if is_bidirectional:
    D = 2
else:
    D = 1

# Define the DataEmbedding class
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
        self.embedding = DataEmbedding(input_dim, embed_dim)  # Data embedding layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=dropout_prob)  # Transformer encoder layer with 8 heads
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # Transformer encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True,
            bidirectional=is_bidirectional
        )  # LSTM layer
        self.decoder = nn.Linear(hidden_dim * D, 1)  # Linear layer for output
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # Create a mask for the Transformer
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.embedding(src)  # Apply embedding
        mask = self._generate_square_subsequent_mask(src.size(1)).to(device)  # Generate mask
        src = src.permute(1, 0, 2)  # Permute dimensions for Transformer
        x = self.transformer_encoder(src, mask)  # Apply Transformer encoder
        x = x.permute(1, 0, 2)  # Permute dimensions back
        hidden_0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)  # Initialize hidden state for LSTM
        c_0 = torch.zeros(num_layers * D, x.size(0), hidden_dim).to(device)  # Initialize cell state for LSTM
        x, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))  # Apply LSTM
        output = self.decoder(x[:, -1, :])  # Apply decoder to the last time step
        return output

# Initialize pth, loss function, and optimizer
TfModel = Transformer().to(device)
optimizer = torch.optim.RMSprop(TfModel.parameters(), lr=learning_rate)  # RMSprop optimizer
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

# Save pth parameters
save_path = '../pth/model .pth'
torch.save(TfModel.state_dict(), save_path)

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
plt.savefig("loss_transformer_bilstm.svg", format='svg')
plt.show()

# Calculate and print evaluation metrics
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
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(timestamps_test, y_true, label='True Values', color='red')
plt.plot(timestamps_test, y_pred, label='Predictions', color='blue')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.ylabel('Actual')
plt.legend(fontsize=12)  # Set legend font size
plt.savefig("prediction_transformer_bilstm.svg", format='svg')
plt.show()

#SHAP analysis (commented out)
background = X_train[:100].to(device)
test_data = X_test[:100].to(device)

# Compute SHAP values using GradientExplainer
TfModel.train()
explainer = shap.GradientExplainer(TfModel, background)
TfModel.eval()
TfModel.train()
shap_values = explainer.shap_values(test_data)
TfModel.eval()
# Get feature names

feature_names = data.columns

# Flatten shap_values to match X_display
shap_values = np.array(shap_values).reshape(-1, X_train.shape[2])
X_display = test_data.cpu().numpy().reshape(-1, X_train.shape[2])

# Visualize explanation for the first prediction
shap.initjs()
force_plot = shap.force_plot(np.mean(shap_values), shap_values[0], X_display[0], feature_names=feature_names)

# Save as HTML file
shap.save_html("force_plot_0.html", force_plot)
# Plot SHAP summary bar chart
shap.summary_plot(shap_values, X_display, feature_names=feature_names, plot_type="bar")
