import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = pd.read_csv('../data_source/continuous dataset.csv')

# Ensure the 'datetime' column is of datetime type
data['datetime'] = pd.to_datetime(data['datetime'])

# Optionally set the 'datetime' column as the index
data.set_index('datetime', inplace=True)
data_target = data['nat_demand']

# Calculate Pearson correlation with the target value
correlations = data.corrwith(data_target).abs()

# Select the top 5 features with the highest correlation
top_5_features = correlations.nlargest(5).index
print(top_5_features)

# Select the top 5 most correlated features
data = data[top_5_features]

# Variables to store processed data
X = []
y = []

# Sliding window size of 7 hours
window_size = 7

# Iterate over the data
for i in range(len(data) - window_size):
    # Extract features within the window
    window_features = data.iloc[i:i + window_size].values
    # Extract target for prediction
    target = data_target.iloc[i + window_size]

    # Append features and target to X and y respectively
    X.append(window_features)
    y.append(target)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Manually split the dataset in order
test_size = 0.1
split_index = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convert to Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset[0])
