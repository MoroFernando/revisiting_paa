import numpy as np
from aeon.datasets import load_classification

def znorm(x): return (x - np.mean(x)) / np.std(x)

def load_and_normalize_dataset(dataset_name):
    print(f'Loading dataset {dataset_name}...')
    X_train, y_train = load_classification(dataset_name, split='train')
    X_test, y_test = load_classification(dataset_name, split='test')
    
    X_train_normalized = np.array([[znorm(series) for series in sample] for sample in X_train])
    X_test_normalized = np.array([[znorm(series) for series in sample] for sample in X_test])
    return X_train_normalized, y_train, X_test_normalized, y_test