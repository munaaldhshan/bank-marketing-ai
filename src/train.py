"""End-to-end training entrypoint: python -m src.train

Downloads the dataset if it's missing, trains the production pipeline, and
saves the model, its metrics, and its feature schema under models/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DATA_PATH, METRICS_PATH, MODEL_PATH
from src.data_loader import download_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default=str(DATA_PATH))
    parser.add_argument('--model-path', default=str(MODEL_PATH))
    args = parser.parse_args(argv)

    # Imported lazily so `--help` doesn't pay for scikit-learn's import time.
    from src.final_pipeline import train_and_save_final_pipeline

    if not Path(args.data_path).exists():
        print(f'Dataset not found at {args.data_path}, downloading...')
        download_dataset(args.data_path)

    print('Training the production pipeline (HistGradientBoosting, duration excluded)...')
    destination = train_and_save_final_pipeline(data_path=args.data_path, model_path=args.model_path)
    print(f'Saved model to {destination}')

    metrics = json.loads(METRICS_PATH.read_text())
    print(f"Decision threshold (tuned on validation split): {metrics['decision_threshold']:.2f}")
    print('Held-out test metrics at that threshold:')
    for name, value in metrics['test_metrics'].items():
        print(f'  {name:20} {value:.4f}')


if __name__ == '__main__':
    main()
