import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
save_dir = Path("data/external")
save_dir.mkdir(parents=True, exist_ok=True)

print("Downloading FRED macro indicators...")
fred = Fred(api_key=FRED_API_KEY)

indicators = {
    "unemployment_rate": "UNRATE",
    "gdp_growth": "A191RL1Q225SBEA",
    "federal_funds_rate": "FEDFUNDS",
    "consumer_price_index": "CPIAUCSL",
    "credit_default_rate": "DRCCLACBS",
}

macro_data = {}
for name, series_id in indicators.items():
    print(f"  Fetching {name}...")
    macro_data[name] = fred.get_series(series_id, observation_start="2010-01-01")

macro_df = pd.DataFrame(macro_data)
macro_df.index.name = "date"
macro_df.to_csv(save_dir / "fred_macro_indicators.csv")
print(f"  Saved {len(macro_df)} rows to data/external/fred_macro_indicators.csv")

print("\nDownloading market signals via yfinance...")
tickers = {
    "SP500": "SPY",
    "VIX": "^VIX",
    "US10Y_yield": "^TNX",
}

market_data = {}
for name, ticker in tickers.items():
    print(f"  Fetching {name}...")
    try:
        df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            market_data[name] = df["Close"]  # type: ignore
            print(f"  Got {len(df)} rows for {name}")  # type: ignore
        else:
            print(f"  Warning: No data for {name}, skipping")
    except Exception as e:
        print(f"  Warning: Could not fetch {name}: {e}, skipping")

market_df = pd.concat(market_data, axis=1)
market_df.columns = list(market_data.keys())
market_df.index.name = "date"
market_df.dropna(how="all", inplace=True)
market_df.to_csv(save_dir / "market_signals.csv")
print(f"  Saved {len(market_df)} rows to data/external/market_signals.csv")

print("\nAll external data downloaded successfully!")
print("\nFiles in data/external/:")
for f in save_dir.iterdir():
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
