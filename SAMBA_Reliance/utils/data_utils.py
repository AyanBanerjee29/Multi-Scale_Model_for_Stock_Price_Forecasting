# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np

# Ensure device is defined at the top level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The MinMaxNorm01 class (This was missing)
class MinMaxNorm01:
    """Scale data to range [0, 1]"""
    
    def __init__(self):
        pass
    
    def fit(self, x):
        # Find min and max along the feature axis (axis=0)
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    
    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min + 1e-8) # Add epsilon for stability
        return x
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        # Ensure max and min are correctly shaped for broadcasting if they are scalars
        max_val = self.max if isinstance(self.max, np.ndarray) else np.array([self.max])
        min_val = self.min if isinstance(self.min, np.ndarray) else np.array([self.min])
        
        # When inverse transforming, we usually only care about the target column (e.g., price)
        # Assuming the first feature [0] is the price. Adjust if not.
        x = x * (max_val[0] - min_val[0] + 1e-8) + min_val[0]
        return x

# 2. The data_loader function (This was also missing)
def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    """Create PyTorch DataLoader from tensors"""
    # X and Y should already be tensors on the correct device from prepare_data
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader

# 3. The corrected prepare_data function
def prepare_data(csv_file, window=60, predict=5):
    """
    Prepare data for training from CSV file for a REGRESSION task.

    Args:
        csv_file: Path to CSV file
        window: Input sequence length (e.g., 60 days)
        predict: Output sequence length (e.g., 5 days)

    Returns:
        XX, YY, mmn (normalizer), num_features
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df.index.name = "Date"

    df.columns = [col.lower() for col in df.columns]

    if 'price' not in df.columns and 'close' in df.columns:
        df.rename(columns={'close': 'price'}, inplace=True)

    if 'price' not in df.columns:
        raise KeyError(f"Fatal Error: Could not find 'price' or 'close' column. Available columns are: {list(df.columns)}")

    # Remove the 'name' column if it exists, as it's non-numeric
    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    df.dropna(inplace=True)

    try:
        price_index = list(df.columns).index('price')
    except ValueError:
        raise KeyError("'price' column not found after processing.")

    data_full = df.to_numpy(dtype=np.float32)

    mmn = MinMaxNorm01()
    dataset = mmn.fit_transform(data_full)
    
    X_seq = []
    Y_seq = []
    
    data_len = len(dataset)
    num_samples = data_len - window - predict + 1

    for i in range(num_samples):
        x_i = dataset[i : i + window, :]
        y_i = dataset[i + window : i + window + predict, price_index]
        
        X_seq.append(x_i)
        Y_seq.append(y_i)

    XX = torch.from_numpy(np.array(X_seq)).float().to(device)
    YY = torch.from_numpy(np.array(Y_seq)).float().to(device)
    
    if YY.dim() == 2:
        YY = YY.unsqueeze(-1)

    num_features = XX.shape[2]

    return XX, YY, mmn, num_features
