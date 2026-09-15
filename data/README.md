# Dataset instructions

This project uses the public Bank Marketing dataset from the UCI Machine Learning Repository
(the same data is also mirrored on Kaggle).

## Option 1: automatic download (recommended)

```bash
python -m src.download_data
```

This fetches `bank-additional-full.csv` directly from the
[official UCI archive](https://archive.ics.uci.edu/dataset/222/bank+marketing) — no account
needed — and saves it as `data/bank_marketing.csv`.

## Option 2: manual download

1. Download from UCI (link above) or from
   [Kaggle](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)
   (a free Kaggle account is required for the Kaggle mirror).
2. Place the CSV in this folder as `data/bank_marketing.csv`.

## Important notes

- The raw dataset is intentionally not committed to GitHub (`.gitignore` excludes `data/*.csv`).
- `src/data_loader.find_dataset_path` also recognizes `bank-additional-full.csv` if you keep the
  original filename.
- `python -m src.train` will download the dataset automatically if it's missing.
