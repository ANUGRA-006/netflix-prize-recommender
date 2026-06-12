#!/usr/bin/env python
"""
Step 1/4 - Data preparation  (standalone)

Loads the raw Netflix Prize files, prints an EDA summary, builds a
signal-preserving subset, and writes a leakage-free per-user time split.

Outputs (in ./artifacts/):
    train.parquet, test.parquet, movies.parquet, data_summary.json

Run:
    python 01_prepare_data.py

Configure by editing the CONFIG block below, or via env vars, e.g.:
    DATA_DIR=/kaggle/input/netflix-prize-data python 01_prepare_data.py
"""
import os
import json
import numpy as np
import pandas as pd

# ============================ CONFIG - edit these ============================
# Folder that holds: combined_data_1.txt ... combined_data_4.txt, movie_titles.csv
DATA_DIR         = os.environ.get("DATA_DIR", "data/raw")
ARTIFACTS_DIR    = os.environ.get("ARTIFACTS_DIR", "artifacts")
SEED             = int(os.environ.get("SEED", 42))
N_TOP_MOVIES     = int(os.environ.get("N_TOP_MOVIES", 2000))   # keep N most-rated movies
N_USERS          = int(os.environ.get("N_USERS", 10000))       # then sample this many users
MIN_USER_RATINGS = int(os.environ.get("MIN_USER_RATINGS", 20)) # min ratings a user must have
TEST_FRACTION    = float(os.environ.get("TEST_FRACTION", 0.20))# hold out each user's most-recent 20%
# =============================================================================


def parse_combined_file(path):
    """Read one combined_data_*.txt (movie-id-as-a-colon-header format)."""
    movie_ids, user_ids, ratings, dates = [], [], [], []
    current_movie = -1
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie = int(line[:-1])
            else:
                u, r, d = line.split(",")
                movie_ids.append(current_movie)
                user_ids.append(int(u))
                ratings.append(int(r))
                dates.append(d)
    return movie_ids, user_ids, ratings, dates


def find_rating_files(data_dir):
    files = [os.path.join(data_dir, f"combined_data_{i}.txt") for i in range(1, 5)]
    found = [f for f in files if os.path.exists(f)]
    if not found:
        raise FileNotFoundError(
            f"No combined_data_*.txt files found in {os.path.abspath(data_dir)}. "
            f"Download the dataset from Kaggle and set DATA_DIR (see README)."
        )
    return found


def load_all_ratings(files):
    frames = []
    for path in files:
        m, u, r, d = parse_combined_file(path)
        df = pd.DataFrame({
            "movie_id": np.asarray(m, dtype=np.int32),
            "user_id":  np.asarray(u, dtype=np.int32),
            "rating":   np.asarray(r, dtype=np.int8),
            "date":     pd.to_datetime(d),
        })
        frames.append(df)
        print(f"  {os.path.basename(path)}: {len(df):,} ratings parsed")
    return pd.concat(frames, ignore_index=True)


def load_movie_titles(path):
    """Parse movie_titles.csv (unquoted, Latin-1, commas inside titles, NULL years)."""
    rows = []
    with open(path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            mid, year, title = line.strip().split(",", 2)
            rows.append((int(mid), None if year == "NULL" else int(year), title))
    return pd.DataFrame(rows, columns=["movie_id", "year", "title"]).set_index("movie_id")


def make_subset(ratings, rng):
    """Keep the most-rated movies + a sample of users who rated enough of them."""
    movie_pop = ratings.groupby("movie_id").size()
    top_movies = movie_pop.sort_values(ascending=False).head(N_TOP_MOVIES).index
    sub = ratings[ratings.movie_id.isin(top_movies)]
    active = sub.groupby("user_id").size()
    eligible = active[active >= MIN_USER_RATINGS].index.to_numpy()
    chosen = rng.choice(eligible, size=min(N_USERS, len(eligible)), replace=False)
    return sub[sub.user_id.isin(chosen)].copy()


def time_split(sub):
    """Per-user, by-date split: hold out each user's most recent TEST_FRACTION."""
    sub = sub.sort_values(["user_id", "date"]).reset_index(drop=True)
    position   = sub.groupby("user_id").cumcount()
    user_total = sub.groupby("user_id")["user_id"].transform("size")
    n_test     = np.maximum((user_total * TEST_FRACTION).astype(int), 1)
    is_test    = position >= (user_total - n_test)
    return sub[~is_test].copy(), sub[is_test].copy()


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"[1/4] Loading raw ratings from {DATA_DIR} ...")
    ratings = load_all_ratings(find_rating_files(DATA_DIR))
    n_users, n_movies = ratings.user_id.nunique(), ratings.movie_id.nunique()
    density = len(ratings) / (n_users * n_movies)
    print(f"  TOTAL: {len(ratings):,} ratings | {n_users:,} users | "
          f"{n_movies:,} movies | density {density:.3%}")

    movies = load_movie_titles(os.path.join(DATA_DIR, "movie_titles.csv"))
    print(f"  {len(movies):,} movie titles loaded "
          f"({int(movies.year.isna().sum())} missing year)")

    print("  Building subset ...")
    sub = make_subset(ratings, rng)
    sub_density = len(sub) / (sub.user_id.nunique() * sub.movie_id.nunique())
    print(f"  Subset: {len(sub):,} ratings | {sub.user_id.nunique():,} users | "
          f"{sub.movie_id.nunique():,} movies | density {sub_density:.2%}")
    del ratings

    print("  Time-splitting (per user, by date) ...")
    train_df, test_df = time_split(sub)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} "
          f"({len(test_df)/len(sub):.1%} held out)")

    train_df.to_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"))
    test_df.to_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))
    movies.to_parquet(os.path.join(ARTIFACTS_DIR, "movies.parquet"))
    summary = {
        "n_ratings": int(len(train_df) + len(test_df)),
        "n_users": int(sub.user_id.nunique()),
        "n_movies": int(sub.movie_id.nunique()),
        "global_mean_rating": round(float(sub.rating.mean()), 4),
        "subset_density": round(sub_density, 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    with open(os.path.join(ARTIFACTS_DIR, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[1/4] Done. Artifacts written to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
