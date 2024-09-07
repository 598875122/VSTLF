import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../continuous dataset.csv')

# Ensure the datetime column is of datetime type
data['datetime'] = pd.to_datetime(data['datetime'])

# Optionally set the datetime column as index
data.set_index('datetime', inplace=True)
data_target = data
# data = data.drop(['nat_demand'], axis=1)

# Variables to store processed data
X = []
y = []
timestamps = []

# Sliding window size of 7 hours
window_size = 7

# Iterate over the data
for i in range(len(data) - window_size):
    # Extract features within the window
    window_features = data.iloc[i:i + window_size].values
    # Extract target for prediction
    target = data_target['nat_demand'].iloc[i + window_size]

    # Save the timestamp
    timestamps.append(data.index[i + window_size])
    # Append features and target to X and y respectively
    X.append(window_features)
    y.append(target)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Manually split the dataset in order
test_size = 0.1
split_index = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
timestamps_train, timestamps_test = timestamps[:split_index], timestamps[split_index:]

# Convert to Tensor
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
