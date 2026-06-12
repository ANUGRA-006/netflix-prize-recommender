#!/usr/bin/env python
"""Step 1/4 — Data preparation.

Loads the raw Netflix Prize files, prints an EDA summary, builds the
signal-preserving subset, and writes a leakage-free per-user time split.

Outputs (in artifacts/):
    train.parquet, test.parquet, movies.parquet, data_summary.json

Run:
    python scripts/01_prepare_data.py            # uses config.yaml
    DATA_DIR=/kaggle/input/netflix-prize-data python scripts/01_prepare_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.config import load_config
from recsys.data import (
    dataset_summary,
    find_rating_files,
    load_all_ratings,
    load_movie_titles,
    make_subset,
    time_split,
)


def main():
    cfg = load_config()
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.seed)

    print(f"[1/4] Loading raw ratings from {cfg.data_dir} ...")
    files = find_rating_files(cfg.data_dir)
    ratings = load_all_ratings(files)
    summary = dataset_summary(ratings)
    print(
        f"  TOTAL: {summary['n_ratings']:,} ratings | {summary['n_users']:,} users "
        f"| {summary['n_movies']:,} movies | density {summary['density']:.3%}"
    )

    movies = load_movie_titles(cfg.data_dir / "movie_titles.csv")
    print(f"  {len(movies):,} movie titles loaded "
          f"({int(movies.year.isna().sum())} missing year)")

    print("  Building subset ...")
    sub = make_subset(
        ratings,
        n_top_movies=cfg.subset.n_top_movies,
        n_users=cfg.subset.n_users,
        min_user_ratings=cfg.subset.min_user_ratings,
        rng=rng,
    )
    sub_density = len(sub) / (sub.user_id.nunique() * sub.movie_id.nunique())
    print(
        f"  Subset: {len(sub):,} ratings | {sub.user_id.nunique():,} users "
        f"| {sub.movie_id.nunique():,} movies | density {sub_density:.2%}"
    )
    del ratings  # free the full table

    print("  Time-splitting (per user, by date) ...")
    train_df, test_df = time_split(sub, test_fraction=cfg.split.test_fraction)
    print(
        f"  Train: {len(train_df):,} | Test: {len(test_df):,} "
        f"({len(test_df) / len(sub):.1%} held out)"
    )

    a = cfg.artifacts_dir
    train_df.to_parquet(a / "train.parquet")
    test_df.to_parquet(a / "test.parquet")
    movies.to_parquet(a / "movies.parquet")
    summary["subset_density"] = round(sub_density, 4)
    summary["n_train"] = int(len(train_df))
    summary["n_test"] = int(len(test_df))
    with open(a / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[1/4] Done. Artifacts written to {a}/")


if __name__ == "__main__":
    main()
