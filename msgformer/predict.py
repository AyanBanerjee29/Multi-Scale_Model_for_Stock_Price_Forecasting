import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data.data_loader import FinancialDataLoader
from models.msgformer import MSGformer
from utils.metrics import calculate_metrics
from utils.preprocessing import denormalize_data


def plot_predictions(y_true, y_pred, ticker, save_path='plots'):
    """Plot prediction results"""
    Path(save_path).mkdir(exist_ok=True)

    plt.figure(figsize=(15, 6))

    # Plot subset for visibility
    plot_samples = min(1000, len(y_true))

    plt.plot(y_true[:plot_samples], label='Actual', alpha=0.7, linewidth=1.5)
    plt.plot(y_pred[:plot_samples], label='Predicted', alpha=0.7, linewidth=1.5)

    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.title(f'{ticker} - Actual vs Predicted Close Price', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'{save_path}/{ticker}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved prediction plot to {save_path}/{ticker}_predictions.png")


def plot_error_distribution(y_true, y_pred, ticker, save_path='plots'):
    """Plot error distribution"""
    Path(save_path).mkdir(exist_ok=True)

    errors = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Error histogram
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    axes[1].set_xlabel('Actual Price', fontsize=12)
    axes[1].set_ylabel('Predicted Price', fontsize=12)
    axes[1].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{ticker}_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved error analysis plot to {save_path}/{ticker}_error_analysis.png")


def predict_on_ticker(model, config, ticker, device):
    """Make predictions for a specific ticker"""
    print(f"\nProcessing {ticker}...")

    # Load data
    data_loader = FinancialDataLoader(config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data(ticker)

    # Make predictions on test set
    model.eval()
    X_test = X_test.to(device)

    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.squeeze(-1).cpu().numpy()

    y_test_np = y_test.cpu().numpy()

    # Denormalize predictions and targets
    predictions_denorm = denormalize_data(predictions, data_loader.scaler, feature_idx=3)
    y_test_denorm = denormalize_data(y_test_np, data_loader.scaler, feature_idx=3)

    # Calculate metrics
    metrics = calculate_metrics(y_test_denorm, predictions_denorm)

    print(f"\nResults for {ticker}:")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
    print(f"  R²: {metrics['R2']:.6f}")

    # Plot results
    plot_predictions(y_test_denorm, predictions_denorm, ticker)
    plot_error_distribution(y_test_denorm, predictions_denorm, ticker)

    return metrics


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = MSGformer(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Make predictions for all tickers
    all_metrics = {}

    for ticker in config['data']['tickers']:
        try:
            metrics = predict_on_ticker(model, config, ticker, device)
            all_metrics[ticker] = metrics
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL TICKERS")
    print("="*70)
    print(f"{'Ticker':<15} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12} {'R²':<10}")
    print("-"*70)

    for ticker, metrics in all_metrics.items():
        print(f"{ticker:<15} {metrics['MAE']:<12.6f} {metrics['RMSE']:<12.6f} "
              f"{metrics['MAPE']:<12.4f} {metrics['R2']:<10.6f}")

    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with MSGformer')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    main(args)

