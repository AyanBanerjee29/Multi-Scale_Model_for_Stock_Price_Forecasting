import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class FinancialDataLoader:
    """Data loader for financial time series from yfinance"""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()

    def download_data(self, ticker: str) -> pd.DataFrame:
        """Download data from yfinance"""
        print(f"Downloading data for {ticker}...")

        data = yf.download(
            ticker,
            start=self.config['data']['start_date'],
            end=self.config['data']['end_date'],
            interval=self.config['data']['interval'],
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        # Select OHLCV features
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.dropna()

        return data

    def create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences"""
        X, y = [], []

        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + pred_len, 3])  # predict Close price

        return np.array(X), np.array(y)

    def prepare_data(self, ticker: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare train, validation and test datasets"""

        # Download data
        data = self.download_data(ticker)

        # Split data
        n = len(data)
        train_size = int(n * self.config['data']['train_ratio'])
        val_size = int(n * self.config['data']['val_ratio'])

        train_data = data[:train_size].values
        val_data = data[train_size:train_size + val_size].values
        test_data = data[train_size + val_size:].values

        # Normalize data
        train_data_scaled = self.scaler.fit_transform(train_data)
        val_data_scaled = self.scaler.transform(val_data)
        test_data_scaled = self.scaler.transform(test_data)

        # Create sequences
        seq_len = self.config['training']['seq_len']
        pred_len = self.config['training']['pred_len']

        X_train, y_train = self.create_sequences(train_data_scaled, seq_len, pred_len)
        X_val, y_val = self.create_sequences(val_data_scaled, seq_len, pred_len)
        X_test, y_test = self.create_sequences(test_data_scaled, seq_len, pred_len)

        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

