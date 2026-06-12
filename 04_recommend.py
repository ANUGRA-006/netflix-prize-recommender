#!/usr/bin/env python
"""
Step 4/4 - Generate recommendations  (standalone)

Prints a Top-N list (with predicted scores), the "because you liked..."
explanations, and a brand-new-user cold-start list. If no user id is given,
picks an arbitrary user from the training set.

Run:
    python 04_recommend.py --user 1621025 --n 10
    python 04_recommend.py                 # arbitrary user
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd

# ============================ CONFIG - edit these ============================
ARTIFACTS_DIR       = os.environ.get("ARTIFACTS_DIR", "artifacts")
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", 3.5))
DAMPING             = int(os.environ.get("DAMPING", 200))   # cold-start popularity prior
# =============================================================================


def all_item_ids(trainset):
    return np.array(list(trainset._raw2inner_id_items.keys()))


def build_user_history(train_df):
    return train_df.groupby("user_id")["movie_id"].apply(set).to_dict()


def recommend_top_n(model, user_id, all_items, history, n=10):
    seen = history.get(user_id, set())
    candidates = all_items[~np.isin(all_items, list(seen))] if seen else all_items
    scores = np.fromiter(
        (model.predict(user_id, int(mid)).est for mid in candidates),
        dtype=float, count=len(candidates),
    )
    order = np.argsort(-scores)[:n]
    return candidates[order], scores[order]


def title_of(movie_id, movies):
    if movie_id in movies.index:
        row = movies.loc[movie_id]
        yr = f" ({int(row.year)})" if pd.notna(row.year) else ""
        return f"{row.title}{yr}"
    return f"Movie {movie_id}"


def damped_popularity(train_df, damping):
    mu = train_df.rating.mean()
    cnt = train_df.groupby("movie_id").size()
    mean = train_df.groupby("movie_id").rating.mean()
    return ((cnt * mean + damping * mu) / (cnt + damping)).sort_values(ascending=False)


def show_recs(user_id, svd, train_df, movies, all_items, history, relevant, k):
    hist = train_df[train_df.user_id == user_id].sort_values("rating", ascending=False)
    print(f"=== User {user_id} - a few movies they rated highly ===")
    for _, r in hist.head(5).iterrows():
        print(f"   {int(r.rating)}*  {title_of(r.movie_id, movies)}")
    items, scores = recommend_top_n(svd, user_id, all_items, history, n=k)
    print(f"--- Top-{k} recommendations ---")
    for rank, (m, s) in enumerate(zip(items, scores), 1):
        hit = "   <-- actually liked in the test set!" if m in relevant else ""
        print(f"  {rank:>2}. {title_of(m, movies)}   [predicted {s:.2f}]{hit}")
    return items


def explain(user_id, svd, knn, trainset, train_df, movies, all_items, history, k=5, n_evidence=2):
    top_items, _ = recommend_top_n(svd, user_id, all_items, history, n=k)
    liked = train_df[(train_df.user_id == user_id) & (train_df.rating >= 4)]
    liked_rating = dict(zip(liked.movie_id.values, liked.rating.values))
    print(f"\n=== Why these are recommended to user {user_id} ===\n")
    for rank, m in enumerate(top_items, 1):
        print(f"{rank}. {title_of(m, movies)}")
        try:
            inner_m = trainset.to_inner_iid(int(m))
        except ValueError:
            inner_m = None
        evidence = []
        if inner_m is not None:
            for lm in liked_rating:
                try:
                    inner_l = trainset.to_inner_iid(int(lm))
                    evidence.append((lm, knn.sim[inner_m, inner_l]))
                except ValueError:
                    pass
        evidence = [(lm, s) for lm, s in sorted(evidence, key=lambda x: -x[1]) if s > 0][:n_evidence]
        if evidence:
            because = " and ".join(f"'{title_of(lm, movies)}' (rated {liked_rating[lm]}*)"
                                   for lm, _ in evidence)
            print(f"     -> because you enjoyed {because}, and similar viewers liked this too.\n")
        else:
            print("     -> matches your overall taste profile (no single strong look-alike).\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=None, help="user id to recommend for")
    ap.add_argument("--n", type=int, default=10, help="number of recommendations")
    args = ap.parse_args()

    md = os.path.join(ARTIFACTS_DIR, "models")
    train_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"))
    test_df  = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))
    movies   = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies.parquet"))
    trainset = joblib.load(os.path.join(md, "trainset.pkl"))
    svd = joblib.load(os.path.join(md, "svd.pkl"))
    knn = joblib.load(os.path.join(md, "knn.pkl"))

    all_items = all_item_ids(trainset)
    history = build_user_history(train_df)
    test_by_user = {}
    for u, m, r in zip(test_df.user_id.values, test_df.movie_id.values, test_df.rating.values):
        test_by_user.setdefault(u, []).append((m, r))

    user = args.user if args.user is not None else int(train_df.user_id.iloc[0])
    relevant = {m for m, r in test_by_user.get(user, []) if r >= RELEVANCE_THRESHOLD}

    show_recs(user, svd, train_df, movies, all_items, history, relevant, args.n)
    explain(user, svd, knn, trainset, train_df, movies, all_items, history, k=min(5, args.n))

    print("=== Cold-start: what a brand-new user would see (damped popularity) ===")
    pop = damped_popularity(train_df, DAMPING)
    for i, m in enumerate(pop.head(args.n).index, 1):
        print(f"  {i:>2}. {title_of(m, movies)}")


if __name__ == "__main__":
    main()
