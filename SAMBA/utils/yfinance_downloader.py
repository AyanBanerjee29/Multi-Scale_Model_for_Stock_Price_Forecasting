import yfinance as yf
import pandas as pd
import os

def download_yfinance_data(ticker, start_date, end_date, output_folder="Dataset"):
    """
    Downloads historical stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., "RELIANCE.NS").
        start_date (str): The start date for the data in "YYYY-MM-DD" format.
        end_date (str): The end date for the data in "YYYY-MM-DD" format.
        output_folder (str): The folder where the CSV file will be saved.

    Returns:
        str: The path to the saved CSV file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date)

    # If yfinance returns multi-index columns, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Standardize all column names to lowercase
    data.columns = [str(col).lower() for col in data.columns]

    # Rename 'close' to 'price' for consistency
    if 'close' in data.columns:
        data.rename(columns={"close": "price"}, inplace=True)

    # Add the 'name' column
    data["name"] = ticker

    # Define the output path
    output_path = os.path.join(output_folder, f"{ticker}.csv")

    # Save to CSV
    data.to_csv(output_path)

    return output_path
