import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc=1, freq='h'):
    """
    `time_features` extracts time features from a `dates` dataframe with a 'dates' column
    down to the specified `freq`. When `timeenc` is 0:
    * m - [month]
    * w - [month]
    * d - [month, day, weekday]
    * b - [month, day, weekday]
    * h - [month, day, weekday, hour]
    * t - [month, day, weekday, hour, minute]

    When `timeenc` is 1, the following `freq` values are supported (all encoded between [-0.5 and 0.5]):
    * Q - [month]
    * M - [month]
    * W - [Day of month, week of year]
    * D - [Day of week, day of month, day of year]
    * B - [Day of week, day of month, day of year]
    * H - [Hour of day, day of week, day of month, day of year]
    * T - [Minute of hour, hour of day, day of week, day of month, day of year]
    * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc == 0:
        dates['month'] = dates.date.apply(lambda row: row.month, 1)
        dates['day'] = dates.date.apply(lambda row: row.day, 1)
        dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
        dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
        dates['minute'] = dates.date.apply(lambda row: row.minute, 1)
        dates['minute'] = dates.minute.map(lambda x: x // 15)
        freq_map = {
            'y': [], 'm': ['month'], 'w': ['month'], 'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'], 'h': ['month', 'day', 'weekday', 'hour'],
            't': ['month', 'day', 'weekday', 'hour', 'minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1, 0)

# Read data
data = pd.read_csv(r'..\continuous dataset.csv')

# Convert Series to DataFrame and rename column to 'date'
df_time = pd.DataFrame({'date': pd.to_datetime(data['datetime'])})

# Now we can call time_features function as we have the correct DataFrame
features = time_features(df_time, timeenc=1, freq='h')

# 1. Separate features and labels
X = data.iloc[:, 1:]  # Other features excluding the first column (time) and the last column (label)
y = data['nat_demand']  # The last column is the label
y = y.to_numpy()

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Combine time features with other features
# X_combined = np.hstack((X_scaled, features))

# Define sliding window
def create_sliding_window_data(X, y, features, window_size=5):
    X_windows = []
    y_windows = []
    feature_windows = []

    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size])
        feature_windows.append(features[i:i + window_size])

    return np.array(X_windows), np.array(y_windows), np.array(feature_windows)

# Create sliding window data
window_size = 7
X_windows, y_windows, feature_windows = create_sliding_window_data(X_scaled, y, features, window_size=window_size)

# 4. Split into training and test sets
# X_train, X_test, y_train, y_test, train_features, test_features = train_test_split(X_windows, y_windows,
#                                                                                    feature_windows, test_size=0.1)

# Manually split dataset in order
test_size = 0.05
split_index = int(len(X_windows) * (1 - test_size))
X_train, X_test = X_windows[:split_index], X_windows[split_index:]
y_train, y_test = y_windows[:split_index], y_windows[split_index:]
train_features, test_features = feature_windows[:-70], feature_windows[split_index:]

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

# 5. Create custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y, data_stamp):
        self.X = X
        self.y = y
        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.data_stamp[idx]

# Create Dataset objects for training and test sets
train_dataset = CustomDataset(X_train_tensor, y_train_tensor, train_features_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor, test_features_tensor)

# 6. Use DataLoader to wrap data
batch_size = 32  # Choose a batch size suitable for your pth and hardware
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f'Features:\n{train_dataset[10][0]}')
print(f'Labels:{train_dataset[10][1]}')
print(f'Time_Features:\n{train_dataset[10][2]}')
