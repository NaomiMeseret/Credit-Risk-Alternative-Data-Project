"""
Processing script to generate model-ready dataset with proxy target.
- Loads raw data (data/raw/data.csv by default)
- Engineers RFM features and proxy label (is_high_risk)
- Saves processed dataset to data/processed/processed.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from src.data_processing import engineer_rfm_features, create_proxy_variable


def process(raw_path: Path, output_path: Path):
    df = pd.read_csv(raw_path)
    rfm = engineer_rfm_features(df, customer_id_col='CustomerId')
    df_labeled = create_proxy_variable(df, rfm_df=rfm, n_clusters=3, random_state=42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path} with {len(df_labeled)} rows")


def main():
    parser = argparse.ArgumentParser(description="Process raw data to create proxy target")
    parser.add_argument('--raw_path', type=str, default='data/raw/data.csv', help='Path to raw CSV')
    parser.add_argument('--output_path', type=str, default='data/processed/processed.csv', help='Output CSV path')
    args = parser.parse_args()
    process(Path(args.raw_path), Path(args.output_path))


if __name__ == "__main__":
    main()
