# -*- coding: utf-8 -*-
"""
Main training and testing script for SAMBA stock price forecasting model
"""
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # Required for getting price_index
from sklearn.model_selection import TimeSeriesSplit
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed,
    pearson_correlation, rank_information_coefficient, All_Metrics, data_loader
)
from utils.yfinance_downloader import download_yfinance_data
from trainer import Trainer

# --- GPU/CPU Configuration ---
# This is the line you will change to enable GPU.
# Set to "cpu" because your GT 710 is not supported by modern PyTorch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# To enable GPU (if you get a compatible one), change the line above to:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------


def masked_mae_loss(scaler, mask_value):
    """Masked MAE loss function (unused, but kept from original)"""
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss


def main(cli_args):
    """Main training and testing function"""
    # Get paper configuration
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print(f"📚 Paper: {dataset_info['paper_title']}")
    print(f"🏛️  Conference: {dataset_info['conference']}")
    print(f"👥 Authors: {', '.join(dataset_info['authors'])}")
    print(f"📊 Expected Features: {dataset_info['total_features']}")
    print("=" * 70)
    print(f"Using device: {device}") # Print which device is being used

    # Initialize seed for reproducibility
    init_seed(config.seed)

    # --- Data Preparation ---
    print("Loading and preparing data...")
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

    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        return

    # Prepare the full dataset
    XX, YY, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
    )

    # --- Get Price Index for Correct Inverse Transform ---
    try:
        df_for_index = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
        df_for_index.columns = [col.lower() for col in df_for_index.columns]
        if 'price' not in df_for_index.columns and 'close' in df_for_index.columns:
            df_for_index.rename(columns={'close': 'price'}, inplace=True)
        if 'name' in df_for_index.columns:
             df_for_index = df_for_index.drop(columns=['name'])
        price_index = list(df_for_index.columns).index('price')
        print(f"Found 'price' column at index: {price_index}")
    except Exception as e:
        print(f"Error finding price index: {e}. Defaulting to index 0.")
        price_index = 0
    
    # Get the min/max for the 'price' column *only*
    price_min = mmn.min[price_index]
    price_max = mmn.max[price_index]
    # ---------------------------------------------------

    config.num_nodes = num_features
    print(f"Number of features (graph nodes): {num_features}")

    # Convert config to dict for compatibility
    args = config.to_dict()

    # --- Cross-Validation Setup ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Create a directory for plots and models
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --- TRAINING MODE ---
    if cli_args.mode == 'train':
        print("\n===== Running in TRAINING Mode =====")
        for fold, (train_index, test_index) in enumerate(tscv.split(XX)):
            print(f"\n===== FOLD {fold + 1}/{n_splits} --- TRAINING =====")

            X_train, X_test = XX[train_index], XX[test_index]
            y_train, y_test = YY[train_index], YY[test_index]

            train_loader = data_loader(X_train, y_train, 64, shuffle=True, drop_last=True)
            val_loader = data_loader(X_test, y_test, 64, shuffle=False, drop_last=False)

            print("Initializing SAMBA model...")
            model_args.vocab_size = num_features # Use actual features
            
            # --- CORRECTED 6-Argument Model Initialization ---
            model = SAMBA(
                model_args,
                args.get('hid'),
                args.get('lag'),
                args.get('horizon'),
                args.get('embed_dim'),
                args.get("cheb_k")
            ).to(device)
            # --------------------------------------------------

            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

            loss = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr_init'))

            lr_scheduler = None
            if args.get('lr_decay'):
                print('Applying learning rate decay.')
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=[int(0.5 * args.get('epochs')), int(0.75 * args.get('epochs'))],
                    gamma=0.1
                )

            trainer = Trainer(
                model, loss, optimizer, train_loader, val_loader, val_loader,
                args=args, lr_scheduler=lr_scheduler
            )

            print("Starting training...")
            _, _ = trainer.train()

            # Save the best model from the trainer
            model_path = os.path.join(output_dir, f"best_model_fold_{fold + 1}.pth")
            print(f"Saving best model for fold {fold + 1} to {model_path}")
            # Save model to CPU to avoid issues when loading on different devices
            torch.save(trainer.model.state_dict(), model_path)

    # --- EVALUATION MODE ---
    # This block runs after 'train' mode or *only* in 'test' mode
    print(f"\n===== Running in EVALUATION Mode ({cli_args.mode}) =====")
    
    all_mae, all_rmse, all_ic, all_ric = [], [], [], []

    for fold in range(n_splits):
        print(f"\n===== FOLD {fold + 1}/{n_splits} --- EVALUATION =====")
        
        # Get the correct test split for this fold
        try:
            train_index, test_index = list(tscv.split(XX))[fold]
        except IndexError:
            print(f"Error: Not enough data for {n_splits} splits.")
            break
            
        # We need to move test data to the device inside the loop
        X_test, y_test = XX[test_index].to(device), YY[test_index].to(device)
        test_loader = data_loader(X_test, y_test, 64, shuffle=False, drop_last=False)

        # Initialize model
        model_args.vocab_size = num_features
        # --- CORRECTED 6-Argument Model Initialization ---
        model = SAMBA(
            model_args,
            args.get('hid'),
            args.get('lag'),
            args.get('horizon'),
            args.get('embed_dim'),
            args.get("cheb_k")
        ).to(device)
        # --------------------------------------------------

        # Load the saved model
        model_path = os.path.join(output_dir, f"best_model_fold_{fold + 1}.pth")
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at {model_path}")
            print("Please run in 'train' mode first to generate the model file.")
            continue
        
        print(f"Loading model from {model_path}...")
        # Load weights, mapping them to the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Replicate trainer.test() logic
        y_pred_list, y_true_list = [], []
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                # Batches are already on the correct device from data_loader
                pred_batch = model(X_batch)
                y_pred_list.append(pred_batch.cpu())
                y_true_list.append(Y_batch.cpu())
        
        y1 = torch.cat(y_pred_list, dim=0)
        y2 = torch.cat(y_true_list, dim=0)

        # --- CORRECTED Manual Inverse Transform ---
        # Squeeze to remove extra dimensions if they exist
        y_p_norm = y1.cpu().numpy().squeeze()
        y_t_norm = y2.cpu().numpy().squeeze()
        
        # Use the specific min/max of the price column
        y_p = y_p_norm * (price_max - price_min) + price_min
        y_t = y_t_norm * (price_max - price_min) + price_min
        # ----------------------------------------------

        # Calculate metrics
        mae, rmse, _ = All_Metrics(torch.tensor(y_p), torch.tensor(y_t), None, None)
        IC = pearson_correlation(torch.tensor(y_t), torch.tensor(y_p))
        RIC = rank_information_coefficient(torch.tensor(y_t).squeeze(), torch.tensor(y_p).squeeze())

        print(f"Fold {fold + 1} Results:")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, IC: {IC:.4f}, RIC: {RIC:.4f}")

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_ic.append(IC)
        all_ric.append(RIC.item() if torch.is_tensor(RIC) else RIC)

        # --- CORRECTED Plotting Logic ---
        # Plot the first day of the horizon (Day-1 prediction)
        plt.figure(figsize=(12, 6))
        # Ensure y_t and y_p are 2D [samples, horizon] before plotting
        if y_t.ndim == 1: y_t = y_t.reshape(-1, 1)
        if y_p.ndim == 1: y_p = y_p.reshape(-1, 1)

        plt.plot(y_t[:, 0], label="Actual (Day 1)", linewidth=2)
        plt.plot(y_p[:, 0], label="Predicted (Day 1)", linewidth=2, linestyle="--")
        plt.xlabel("Time (Test Samples)")
        plt.ylabel("Stock Price")
        plt.title(f"SAMBA Stock Prediction - Fold {fold + 1} (Day-1 Forecast)")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"test_plot_fold_{fold + 1}.png")
        plt.savefig(plot_path)
        print(f"Test plot for fold {fold + 1} saved to {plot_path}")
        plt.close() # Close plot to save memory

    # --- Final Results ---
    if all_mae:
        print("\n===== Cross-Validation Final Results =====")
        print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
        print(f"Average RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
        print(f"Average IC: {np.mean(all_ic):.4f} ± {np.std(all_ic):.4f}")
        print(f"Average RIC: {np.mean(all_ric):.4f} ± {np.std(all_ric):.4f}")
    else:
        print("\nNo evaluation results to display (e.g., if ran in 'test' mode without model files).")

    print("\nScript finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAMBA Model Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode to run: "train" (trains and then tests) or "test" (only tests)')
    cli_args = parser.parse_args()
    
    # Pass parsed arguments to main
    main(cli_args)
