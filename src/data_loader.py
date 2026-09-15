"""Loading, validating and downloading the bank marketing dataset."""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd

from src.config import DATA_DIR, DATA_PATH, DATASET_URL, FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET

# The repo's suggested name first, then the original UCI/Kaggle file name.
CANDIDATE_FILENAMES = ('bank_marketing.csv', 'bank-additional-full.csv', 'bank-marketing.csv')
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, *LEAKAGE_COLUMNS, TARGET)


def find_dataset_path(data_dir: str | Path = DATA_DIR) -> Path:
    """Locate the dataset in data/ under any of its common names."""
    for name in CANDIDATE_FILENAMES:
        path = Path(data_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f'No dataset found in {data_dir}. Run `python -m src.download_data`, or download '
        'bank-additional-full.csv from UCI or Kaggle into data/.'
    )


def read_customer_csv(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read a semicolon- or comma-delimited CSV from a path or an uploaded file."""
    if hasattr(source, 'getvalue'):
        text = source.getvalue().decode('utf-8-sig')
    else:
        text = Path(source).read_text(encoding='utf-8-sig')
    header = text.split('\n', 1)[0]
    separator = ';' if header.count(';') > header.count(',') else ','
    return pd.read_csv(io.StringIO(text), sep=separator)


def load_bank_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the labelled dataset and check that it has the target column."""
    df = read_customer_csv(path if path is not None else find_dataset_path())
    if TARGET not in df.columns:
        raise ValueError(f"The dataset does not contain the target column '{TARGET}'.")
    return df


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Expected columns that are absent (e.g. when the older bank-full.csv is used)."""
    return [col for col in EXPECTED_COLUMNS if col not in df.columns]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_dataset(destination: str | Path = DATA_PATH, url: str = DATASET_URL) -> Path:
    """Download bank-additional-full.csv from the UCI repository."""
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    # The UCI archive nests bank-additional.zip inside bank+marketing.zip.
    inner = zipfile.ZipFile(io.BytesIO(outer.read('bank-additional.zip')))
    member = next(
        name for name in inner.namelist()
        if name.endswith('bank-additional-full.csv') and not name.startswith('__MACOSX')
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(inner.read(member))
    return destination
