import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_data(data: np.ndarray, scaler: StandardScaler = None) -> tuple:
    """
    Normalize data using StandardScaler

    Args:
        data: Input data
        scaler: Pre-fitted scaler (optional)

    Returns:
        Normalized data and scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)

    return normalized_data, scaler


def denormalize_data(data: np.ndarray, scaler: StandardScaler, feature_idx: int = 3) -> np.ndarray:
    """
    Denormalize data back to original scale

    Args:
        data: Normalized data
        scaler: Fitted scaler
        feature_idx: Index of the feature to denormalize (default: 3 for Close price)

    Returns:
        Denormalized data
    """
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((data.shape[0], scaler.n_features_in_))
    dummy[:, feature_idx] = data.flatten()

    # Inverse transform
    denormalized = scaler.inverse_transform(dummy)

    return denormalized[:, feature_idx]

