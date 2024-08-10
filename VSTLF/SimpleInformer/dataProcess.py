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
    """Hour of day encoded as value between [-0.5, 0.5]"""

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
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    >
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

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

# 读取数据
data = pd.read_csv(r'..\continuous dataset.csv')

# 将 Series 转换为 DataFrame，并重命名列为 'date'
df_time = pd.DataFrame({'date': pd.to_datetime(data['datetime'])})


# 现在可以调用 time_features 函数，因为我们有了正确的 DataFrame
features = time_features(df_time, timeenc=1, freq='h')

# 1. 分离特征和标签
X = data.iloc[:, 1:]  # 除去第一列时间和最后一列标签的其他特征
y = data['nat_demand']  # 最后一列是标签
y = y.to_numpy()
# 2. 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3. 将时间特征与其他特征合并
# X_combined = np.hstack((X_scaled, features))


# 定义滑动窗口
def create_sliding_window_data(X, y, features, window_size=5):
    X_windows = []
    y_windows = []
    feature_windows = []

    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size])
        feature_windows.append(features[i:i + window_size])

    return np.array(X_windows), np.array(y_windows), np.array(feature_windows)


# 创建滑动窗口数据
window_size = 7
X_windows, y_windows, feature_windows = create_sliding_window_data(X_scaled, y, features, window_size=window_size)

# # 4. 划分训练集和测试集
# X_train, X_test, y_train, y_test, train_features, test_features = train_test_split(X_windows, y_windows,
#                                                                                    feature_windows, test_size=0.1)

# 手动按顺序分割数据集
test_size = 0.05
split_index = int(len(X_windows) * (1 - test_size))
X_train, X_test = X_windows[:split_index], X_windows[split_index:]
y_train, y_test = y_windows[:split_index], y_windows[split_index:]
train_features, test_features = feature_windows[:-70], feature_windows[split_index:]

# 将 NumPy 数组转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)


# 5. 创建自定义的 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, X, y, data_stamp):
        self.X = X
        self.y = y
        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.data_stamp[idx]


# 创建训练集和测试集的 Dataset 对象
train_dataset = CustomDataset(X_train_tensor, y_train_tensor, train_features_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor, test_features_tensor)

# 6. 使用 DataLoader 封装数据
batch_size = 32  # 选择一个适合你模型和硬件的批大小
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f'Features:\n{train_dataset[10][0]}')
print(f'Labels:{train_dataset[10][1]}')
print(f'Time_Features:\n{train_dataset[10][2]}')