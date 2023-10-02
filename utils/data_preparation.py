import pandas as pd
import numpy as np

def custom_train_test_split(X, y, test_size=0.2, stratify=None, random_seed=None):
    """ Custom train-test split function that supports stratification for target class """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
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

