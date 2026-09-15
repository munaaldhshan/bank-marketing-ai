"""Fetch the dataset into data/: python -m src.download_data [--force]"""

from __future__ import annotations

import argparse

from src.config import DATA_PATH, DATASET_URL
from src.data_loader import download_dataset, load_bank_data, missing_columns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Download the UCI bank-additional-full dataset.')
    parser.add_argument('--force', action='store_true', help='re-download even if the file exists')
    args = parser.parse_args(argv)

    if DATA_PATH.exists() and not args.force:
        print(f'Dataset already present at {DATA_PATH} (use --force to re-download).')
        return

    print(f'Downloading {DATASET_URL} ...')
    path = download_dataset(DATA_PATH)
    df = load_bank_data(path)
    if missing_columns(df):
        raise SystemExit(f'Unexpected schema, missing: {missing_columns(df)}')
    print(f'Saved {path} ({df.shape[0]:,} rows x {df.shape[1]} columns)')


if __name__ == '__main__':
    main()
