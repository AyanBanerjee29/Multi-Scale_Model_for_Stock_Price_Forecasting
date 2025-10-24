# -*- coding: utf-8 -*-
"""
Main training script for SAMBA stock price forecasting model
"""

import os
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics, data_loader
)
from utils.yfinance_downloader import download_yfinance_data
from trainer import Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mae_loss(scaler, mask_value):
    """Masked MAE loss function"""
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss


def main():
    """Main training function using paper configuration"""
    # Get paper configuration
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print(f"📚 Paper: {dataset_info['paper_title']}")
    print(f"🏛️  Conference: {dataset_info['conference']}")
    print(f"👥 Authors: {', '.join(dataset_info['authors'])}")
    print(f"📊 Expected Features: {dataset_info['total_features']}")
    print("=" * 70)

    # Initialize seed for reproducibility
    init_seed(config.seed)

    # Prepare data
    print("Loading and preparing data...")

    # Download Reliance Industries data
    print("Downloading Reliance Industries stock data...")
    reliance_ticker = "RELIANCE.NS"
    reliance_start_date = "2010-01-01"
    reliance_end_date = "2023-12-31"
    dataset_file = download_yfinance_data(
        ticker=reliance_ticker,
        start_date=reliance_start_date,
        end_date=reliance_end_date,
        output_folder="Dataset"
    )
    print(f"Data downloaded and saved to {dataset_file}")

    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        return

    # Prepare the full dataset
    XX, YY, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
    )

    # Update config with actual number of features (nodes in the graph)
    config.num_nodes = num_features
    print(f"Number of features (graph nodes): {num_features}")

    # Convert config to dict for compatibility
    args = config.to_dict()

    # --- Walk-Forward Cross-Validation ---
    n_splits = 5  # You can adjust the number of splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_mae, all_rmse, all_ic, all_ric = [], [], [], []

    for fold, (train_index, test_index) in enumerate(tscv.split(XX)):
        print(f"\n===== FOLD {fold + 1}/{n_splits} =====")

        # Split data for the current fold
        X_train, X_test = XX[train_index], XX[test_index]
        y_train, y_test = YY[train_index], YY[test_index]

        # Create data loaders
        train_loader = data_loader(X_train, y_train, 64, shuffle=True, drop_last=True)
        test_loader = data_loader(X_test, y_test, 64, shuffle=False, drop_last=False)

        # Initialize model with paper configuration
        print("Initializing SAMBA model...")
        model_args.vocab_size = num_features
        model = SAMBA(
            model_args,
            args.get('hid'),
            args.get('lag'),
            args.get('horizon'),
            args.get('embed_dim'),
            args.get("cheb_k")
        ).to(device)

        # Initialize model parameters
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        # Setup loss function and optimizer
        loss = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr_init'))

        # Setup learning rate scheduler
        lr_scheduler = None
        if args.get('lr_decay'):
            print('Applying learning rate decay.')
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[int(0.5 * args.get('epochs')), int(0.75 * args.get('epochs'))],
                gamma=0.1
            )

        # Initialize trainer
        trainer = Trainer(
            model, loss, optimizer, train_loader, test_loader, test_loader,
            args=args, lr_scheduler=lr_scheduler
        )

        # Start training
        print("Starting training...")
        _, _ = trainer.train()

        # Evaluate on test set
        print("Evaluating on test set...")
        y1, y2 = trainer.test(trainer.model, trainer.args, test_loader, trainer.logger)

        # Inverse transform to original scale
        y_p = mmn.inverse_transform(y1.cpu().numpy().squeeze())
        y_t = mmn.inverse_transform(y2.cpu().numpy().squeeze())

        # Calculate metrics
        mae, rmse, _ = All_Metrics(torch.tensor(y_p), torch.tensor(y_t), None, None)
        IC = pearson_correlation(torch.tensor(y_t), torch.tensor(y_p))
        RIC = rank_information_coefficient(torch.tensor(y_t).squeeze(), torch.tensor(y_p).squeeze())

        print(f"Fold {fold + 1} Results:")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, IC: {IC:.4f}, RIC: {RIC:.4f}")

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_ic.append(IC)
        all_ric.append(RIC.item())

        # --- Plotting and Saving ---
        plt.figure(figsize=(12, 6))
        plt.plot(y_t, label="Actual", linewidth=2)
        plt.plot(y_p, label="Predicted", linewidth=2, linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title(f"SAMBA Stock Prediction - Fold {fold + 1}")
        plt.legend()
        plt.grid(True)

        # Create a directory for plots if it doesn't exist
        plot_dir = "test_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"test_plot_fold_{fold + 1}.png")
        plt.savefig(plot_path)
        print(f"Test plot for fold {fold + 1} saved to {plot_path}")
        plt.close()  # Close the plot to avoid displaying it in a loop

    # --- Final Results ---
    print("\n===== Cross-Validation Final Results =====")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
    print(f"Average IC: {np.mean(all_ic):.4f} ± {np.std(all_ic):.4f}")
    print(f"Average RIC: {np.mean(all_ric):.4f} ± {np.std(all_ric):.4f}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
