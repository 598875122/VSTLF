import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import shap

# Load the dataset
data = pd.read_csv('../continuous dataset.csv')

# Ensure the 'datetime' column is of datetime type
data['datetime'] = pd.to_datetime(data['datetime'])

# Set 'datetime' column as the index (optional)
data.set_index('datetime', inplace=True)
data_target = data
data = data.drop(['nat_demand'], axis=1)

# Create lists to store processed data
X = []
y = []
timestamps = []

# Sliding window size of 7 hours
window_size = 7

# Iterate through the dataset
for i in range(len(data) - window_size):
    # Extract features from the window
    window_features = data.iloc[i:i + window_size].values
    # Extract the target to be predicted
    target = data_target['nat_demand'].iloc[i + window_size]

    # Save the timestamp
    timestamps.append(data.index[i + window_size])
    # Add features and targets to X and y
    X.append(window_features)
    y.append(target)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split the data into training and testing sets
test_size = 0.1
split_index = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
timestamps_train, timestamps_test = timestamps[:split_index], timestamps[split_index:]

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset[0])

# Model parameters
is_bidirectional = True
num_layers = 1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if is_bidirectional:
    D = 2
else:
    D = 1

# Define the LSTM pth
class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size=X_train.shape[2], hidden_size=256, num_layers=num_layers, batch_first=True, dropout=0.1, bidirectional=is_bidirectional)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        h_0 = torch.zeros(D*num_layers, x.size(0), 256).to(device)
        c_0 = torch.zeros(D*num_layers, x.size(0), 256).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Set the device and initialize pth, loss function, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LstmModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Train the pth
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Evaluate the pth
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# Set font style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='orange')
plt.plot(val_losses, label='Validation Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=12)
plt.savefig("loss_transformer_lstm.svg", format='svg')
plt.show()

# Calculate evaluation metrics
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

# Plot the comparison between true and predicted values
plt.figure(figsize=(10, 6))
plt.plot(timestamps_test, y_true, label='True Values', color='red')
plt.plot(timestamps_test, y_pred, label='Predictions', color='blue')
plt.xticks(rotation=45)
plt.ylabel('Actual')
plt.legend(fontsize=12)
plt.savefig("prediction_transformer_lstm.svg", format='svg')
plt.show()
