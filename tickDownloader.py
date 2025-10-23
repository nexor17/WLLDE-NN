import yfinance as yf
import pandas as pd

def download_real_data(ticker="^GSPC", interval="4h", start_date="2024-01-01", end_date="2025-01-01"):

    # Download OHLCV data
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        prepost=False,
        progress=False
    )

    if data.empty:
        print("No data retrieved.")
        return

    # Flatten multi-index columns if present (removes ticker name from column headers)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index (Datetime as column)
    data.reset_index(inplace=True)
    
    # Drop any rows with NaN or non-numeric values in OHLCV columns
    data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    # Rename datetime column consistently
    data.rename(columns={"Date": "Datetime"}, inplace=True)

    # Format datetime
    data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Reorder columns
    data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Round OHLC prices
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].round(3)

    # Build filename with data directory
    import os
    os.makedirs('data', exist_ok=True)
    filename = f"data/{ticker.replace('^','').replace('.','_')}_{interval}.csv"

    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Saved {len(data)} rows to {filename}")


if __name__ == "__main__":

    # Change these values to control what you download
    ticker = "^GSPC"#input('ticker: ')        # e.g., "QQQ", "SPY", "IWM", "^GSPC"
    interval = "2m"           # options: "1m"(7d), "2m"(60d), "5m"(60d), "15m"(60d), "30m"(60d), "1h"(730d), "1d"(max)
    start_date = "2025-09-01"  # Keep within 50-55 days for safety
    end_date = "2025-10-23"

    download_real_data(ticker, interval, start_date, end_date)
