import pandas as pd
import numpy as np

def custom_train_test_split(X, y, test_size=0.2, stratify=None, random_seed=None):
    """ Custom train-test split function that supports stratification for target class """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # stratification for maintaining target class distribution in train and test sets
    if stratify is not None:
        class_idx = np.where(stratify)[0]
        np.random.shuffle(class_idx)

        test_count = int(test_size * class_idx.shape[0])
        test_indices = class_idx[:test_count]
        train_indices = class_idx[test_count:]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        test_count = int(test_size * X.shape[0])
        train_indices = indices[test_count:]
        test_indices = indices[:test_count]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def pre_processings(train_data, test_data, scale=True, feature_range=(0,1)):
    """Remove outliers and normalize data to feature range"""
    if scale:
        train_data, test_data = min_max_scale(train_data, test_data, feature_range=feature_range)

    return train_data, test_data


def remove_outliers(df):
    """Remove outliers using IQR"""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


def min_max_scale(train_data, test_data, feature_range=(0,1)):
    """Normalize data to feature range using min-max scaling"""
    scale_min, scale_max = feature_range
    min_train = train_data.min()
    max_train = train_data.max()

    train_data_scaled= scale_min + (train_data - min_train) * (scale_max - scale_min) / (max_train - min_train)
    test_data_scaled = scale_min + (test_data - min_train) * (scale_max - scale_min) / (max_train - min_train)   # use min & max from training data
    
    return train_data_scaled, test_data_scaled