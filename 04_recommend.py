#!/usr/bin/env python
"""Step 4/4 — Generate recommendations for a user.

Prints a Top-N list (with predicted scores), the "because you liked..."
explanations, and a brand-new-user cold-start list. If no user id is given,
picks an arbitrary user from the training set.

Run:
    python scripts/04_recommend.py --user 1621025 --n 10
    python scripts/04_recommend.py                 # arbitrary user
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.coldstart import damped_popularity
from recsys.config import load_config
from recsys.evaluate import build_test_index
from recsys.explain import explain_recommendations
from recsys.recommend import all_item_ids, build_user_history, format_user_recs, title_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=None, help="user id to recommend for")
    ap.add_argument("--n", type=int, default=10, help="number of recommendations")
    args = ap.parse_args()

    cfg = load_config()
    a = cfg.artifacts_dir
    train_df = pd.read_parquet(a / "train.parquet")
    test_df = pd.read_parquet(a / "test.parquet")
    movies = pd.read_parquet(a / "movies.parquet")

    trainset = joblib.load(cfg.models_dir / "trainset.pkl")
    svd = joblib.load(cfg.models_dir / "svd.pkl")
    knn = joblib.load(cfg.models_dir / "knn.pkl")

    all_items = all_item_ids(trainset)
    history = build_user_history(train_df)
    test_by_user = build_test_index(test_df)

    user = args.user if args.user is not None else int(train_df.user_id.iloc[0])
    relevant = {m for m, r in test_by_user.get(user, []) if r >= cfg.eval.relevance_threshold}

    print(format_user_recs(
        user, svd, train_df=train_df, movies=movies, all_items=all_items,
        history=history, relevant=relevant, k=args.n,
    ))
    print("\n" + explain_recommendations(
        user, svd=svd, knn=knn, trainset=trainset, train_df=train_df,
        movies=movies, all_items=all_items, history=history,
        k=min(5, args.n),
    ))

    print("=== Cold-start: what a brand-new user would see (damped popularity) ===")
    pop = damped_popularity(train_df, damping=cfg.coldstart.damping)
    for i, m in enumerate(pop.head(args.n).index, 1):
        print(f"  {i:>2}. {title_of(m, movies)}")


if __name__ == "__main__":
    main()
