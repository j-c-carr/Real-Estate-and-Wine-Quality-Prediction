import pandas as pd
import numpy as np


def remove_outliers(df, factor=1.5):
    """Remove outliers using IQR"""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - factor * IQR)) | (df > (Q3 + factor * IQR))).any(axis=1)]


def min_max_scale(train_data, test_data, feature_range=(0,1)):
    """Normalize data to feature range using min-max scaling"""
    scale_min, scale_max = feature_range
    min_train = train_data.min()
    max_train = train_data.max()

    train_data_scaled= scale_min + (train_data - min_train) * (scale_max - scale_min) / (max_train - min_train)
    test_data_scaled = scale_min + (test_data - min_train) * (scale_max - scale_min) / (max_train - min_train)   # use min & max from training data
    
    return train_data_scaled, test_data_scaled


def k_fold_split(X, y, n_splits=5):
    """Split data into k folds for cross-validation"""
    fold_sizes = (len(y) // n_splits) * np.ones(n_splits, dtype=int)
    fold_sizes[:len(y) % n_splits] += 1
    current_idx = 0
    for fold_size in fold_sizes:
        start_idx, stop_idx = current_idx, current_idx + fold_size
        train_idc = list(range(0, start_idx)) + list(range(stop_idx, len(y)))
        test_idc = list(range(start_idx, stop_idx))
        yield X[train_idc], y[train_idc], X[test_idc], y[test_idc]
        current_idx = stop_idx