from VSTLF.data_preprocess.data_raw16 import train_loader, test_loader, X_train
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# Parameters
input_dim = X_train.shape[2]
n_seq = 7
batch_size = 32
output_dim = 1
hidden_dim = 256
embed_dim = 256
n_epochs = 200
num_layers = 1
learning_rate = 1e-3
weight_decay = 1e-6
is_bidirectional = False
dropout_prob = 0.1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if is_bidirectional:
    D = 2
else:
    D = 1

# Define RNN pth
class rnnModel(nn.Module):
    def __init__(self):
        super(rnnModel, self).__init__()

        # RNN or BiRNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=is_bidirectional,
            dropout=dropout_prob,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * D, output_dim)

    def forward(self, x):
        # Initialize hidden state
        hidden_0 = torch.zeros(D * num_layers, x.size(0), hidden_dim).to(device)

        # Forward pass through RNN
        output, h_n = self.rnn(x, hidden_0.detach())

        # Pass the output of the last time step through the fully connected layer
        output = self.fc(output[:, -1, :])

        return output

# Initialize pth, loss function, and optimizer
rnnNet = rnnModel().to(device)
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.RMSprop(rnnNet.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the pth
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Clear gradients
        outputs = rnnNet(inputs)  # Forward pass
        loss = criterion(outputs.squeeze(), targets)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        running_loss += loss.item()  # Accumulate loss
    train_losses.append(running_loss / len(train_loader))  # Average loss per epoch

    print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}')

# Plot loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_rnnNet.svg", format='svg')
plt.show()

# Model prediction and evaluation
rnnNet.eval()  # Set pth to evaluation mode
y_pred = []
y_true = []

with torch.no_grad():  # Disable gradient calculation
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = rnnNet(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())  # Collect predictions
        y_true.extend(targets.cpu().numpy())  # Collect ground truth

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

# Plot comparison of true values and predictions
plt.figure(figsize=(15, 5))
plt.plot(y_true, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='red', linestyle='dashed')
plt.title('Comparison of True Values and Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.savefig("prediction_rnnNet.svg", format='svg')
plt.show()
