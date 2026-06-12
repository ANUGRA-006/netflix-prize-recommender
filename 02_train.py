#!/usr/bin/env python
"""Step 2/4 — Model training.

Loads the prepared training split, builds the Surprise trainset, trains the
three models, and saves them.

Outputs (in artifacts/models/):
    trainset.pkl, baseline.pkl, knn.pkl, svd.pkl

Run:
    python scripts/02_train.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.config import load_config
from recsys.models import build_baseline, build_knn, build_svd, build_trainset


def main():
    cfg = load_config()
    cfg.ensure_dirs()

    train_df = pd.read_parquet(cfg.artifacts_dir / "train.parquet")
    print(f"[2/4] Building trainset from {len(train_df):,} training ratings ...")
    trainset = build_trainset(train_df)
    print(
        f"  Surprise trainset: {trainset.n_users:,} users, "
        f"{trainset.n_items:,} movies, {trainset.n_ratings:,} ratings"
    )

    md = cfg.models_dir
    joblib.dump(trainset, md / "trainset.pkl")

    print("  Training Baseline (bias only) ...")
    joblib.dump(build_baseline(trainset, cfg.models.baseline), md / "baseline.pkl")

    print("  Training Item-based KNN ...")
    joblib.dump(build_knn(trainset, cfg.models.knn), md / "knn.pkl")

    print("  Training SVD ...")
    joblib.dump(build_svd(trainset, cfg.models.svd, cfg.seed), md / "svd.pkl")

    print(f"[2/4] Done. Models saved to {md}/")


if __name__ == "__main__":
    main()
