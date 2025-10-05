# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MinMaxNorm01:
    """Scale data to range [0, 1]"""
    
    def __init__(self):
        pass
    
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    
    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        x = x * (self.max[0] - self.min[0]) + self.min[0]
        return x


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    """Create PyTorch DataLoader from tensors"""
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader


def prepare_data(csv_file, window=5, predict=1, test_ratio=0.15, val_ratio=0.05):
    """
    Prepare data for training from CSV file
    
    Args:
        csv_file: Path to CSV file
        window: Input sequence length
        predict: Output sequence length
        test_ratio: Ratio of data for testing
        val_ratio: Ratio of data for validation
    
    Returns:
        train_loader, val_loader, test_loader, mmn (normalizer)
    """
    # Load data
    X = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
    
    # Basic preprocessing
    name = X["Name"][0]
    del X["Name"]
    cols = X.columns
    X["Target"] = (X["Price"].pct_change().shift(-1) > 0).astype(int)
    X.dropna(inplace=True)
    
    # Convert to numpy
    a = X.to_numpy()
    
    # Normalize data
    mmn = MinMaxNorm01()
    data = a
    dataset = mmn.fit_transform(data)
    
    # Create sequences
    ran = data.shape[0]
    i = 0
    X_seq = []
    Y_seq = []
    
    price_index = list(X.columns).index('Price')

    while i + window < ran:
        X_seq.append(torch.Tensor(dataset[i:i+window, :]))
        Y_seq.append(torch.Tensor(dataset[i+window:i+window+predict, price_index]))
        i += 1
    
    XX = torch.stack(X_seq, dim=0)
    YY = torch.stack(Y_seq, dim=0)
    YY = YY[:, :, None]
    
    # Split data
    test_len = int(test_ratio * XX.shape[0])
    val_len = int(val_ratio * XX.shape[0])
    train_len = XX.shape[0] - test_len - val_len
    
    # Create tensors
    X_test = torch.Tensor.float(XX[:test_len, :, :]).to(device)
    Y_test = torch.Tensor.float(YY[:test_len, :, :]).to(device)
    
    X_train = torch.Tensor.float(XX[test_len:test_len+train_len, :, :]).to(device)
    Y_train = torch.Tensor.float(YY[test_len:test_len+train_len, :, :]).to(device)
    
    X_val = torch.Tensor.float(XX[-val_len:, :, :]).to(device)
    Y_val = torch.Tensor.float(YY[-val_len:, :, :]).to(device)
    
    # Create data loaders
    train_loader = data_loader(X_train, Y_train, 64, shuffle=False, drop_last=False)
    val_loader = data_loader(X_val, Y_val, 64, shuffle=False, drop_last=False)
    test_loader = data_loader(X_test, Y_test, 64, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, mmn, XX.shape[2]
