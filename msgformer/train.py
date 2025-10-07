import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data.data_loader import FinancialDataLoader, TimeSeriesDataset
from models.msgformer import MSGformer
from utils.metrics import calculate_metrics
from utils.preprocessing import denormalize_data


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc="Training")):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(X)
        loss = criterion(predictions.squeeze(-1), y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validating"):
            X, y = X.to(device), y.to(device)

            predictions = model(X)
            loss = criterion(predictions.squeeze(-1), y)

            total_loss += loss.item()

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = calculate_metrics(all_targets, all_preds)

    return total_loss / len(dataloader), metrics


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data_loader = FinancialDataLoader(config)

    # Use first ticker for training
    ticker = config['data']['tickers'][0]
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data(ticker)

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Initialize model
    print("Initializing model...")
    model = MSGformer(config).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Metrics - MAE: {val_metrics['MAE']:.6f}, RMSE: {val_metrics['RMSE']:.6f}, "
              f"MAPE: {val_metrics['MAPE']:.4f}%, R2: {val_metrics['R2']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, 'checkpoints/best_model.pt')
            print("Saved best model!")
        else:
            patience_counter += 1

        if patience_counter >= config['training']['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model and evaluate on test set
    print("\nLoading best model for testing...")
    checkpoint = torch.load('checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"MAE: {test_metrics['MAE']:.6f}")
    print(f"RMSE: {test_metrics['RMSE']:.6f}")
    print(f"MAPE: {test_metrics['MAPE']:.4f}%")
    print(f"R²: {test_metrics['R2']:.6f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MSGformer model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    main(args)

