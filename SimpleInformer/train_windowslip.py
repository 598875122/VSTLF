import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from main_Informer import Informer
from dataProcess import train_loader, test_loader, X_train
from torchinfo import summary

# Model parameters
enc_in = X_train.shape[2]  # Dimension of input features
print(enc_in)
dec_in = enc_in  # Decoder input dimension is the same as encoder
c_out = 1  # Dimension of output features (typically 1 for regression tasks)
seq_len = 7  # Length of the sequence; 1-step prediction in this case
label_len = 1  # Length of the label; 1 for 1-step prediction
out_len = 1  # Length of the output sequence; typically 1 for 1-step prediction
e_layers = 1  # Number of encoder layers
d_layers = 1  # Number of decoder layers
hidden_dim = 512  # Hidden dimension of the feedforward network
n_heads = 4  # Number of attention heads
dropout = 0.1  # Dropout probability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Instantiate the pth
model = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len, e_layers,
                 d_layers, hidden_dim, n_heads, dropout).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

losses = []
# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets, time_f) in enumerate(train_loader):
        # Move data to the device
        inputs, targets, time_f = inputs.to(device), targets.to(device), time_f.to(device)
        # print(inputs.shape, targets.shape, time_f.shape)
        optimizer.zero_grad()
        outputs = model(inputs, inputs, time_f)  # For 1-step prediction, use input data as both encoder and decoder inputs

        outputs = outputs.squeeze(-1)  # Remove the last dimension to shape [batch_size]
        targets = targets.squeeze(-1)  # Ensure target shape is also [batch_size]
        loss = criterion(outputs, targets)  # Loss function does not require extra dimensions
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot loss curve
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Informer Training Loss Curve')
plt.legend()
plt.savefig("Loss_Informer1.svg", format='svg')
plt.show()

# Switch the pth to evaluation mode
model.eval()

# Store true values and predictions
true_values = []
predicted_values = []

# Avoid gradient calculations to speed up computations
with torch.no_grad():
    for inputs, labels, time_f in test_loader:
        # Move data to the device
        inputs, labels, time_f = inputs.to(device), labels.to(device), time_f.to(device)
        outputs = model(inputs, inputs, time_f)  # Decoder input is simplified
        true_values.extend(labels.cpu().numpy())  # Ensure conversion from GPU to CPU, then to numpy
        y_pred = outputs.squeeze().cpu().numpy()  # Ensure conversion from GPU to CPU, then to numpy
        predicted_values.extend(y_pred)

# Compute evaluation metrics
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

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Add DejaVu Sans to support negative signs
# Plot comparison of true values and predictions
plt.figure(figsize=(18, 8))
plt.plot(y_true, label='True Values', color='r', linestyle='-', marker='o', markersize=4)
plt.plot(y_pred, label='Predictions', color='y', linestyle='--', marker='x', markersize=4)
plt.title('Informer Comparison of True Values and Predictions', fontsize=20)
plt.ylabel('Charge (Ah)', fontsize=16)
# Set font size for axis ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(True)
plt.legend()
plt.savefig("prediction_Informer1.svg", format='svg')
plt.show()

# Convert numpy array to DataFrame
df = pd.DataFrame(losses, columns=["Informer"])
# Save to CSV file
df.to_csv('loss_Informer.csv', index=False)

# Convert numpy array to DataFrame
df = pd.DataFrame(y_pred, columns=["Informer"])

# Save to CSV file
df.to_csv('pred_Informer.csv', index=False)
