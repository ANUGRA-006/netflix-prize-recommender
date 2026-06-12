#!/usr/bin/env python
"""
Step 2/4 - Model training  (standalone)

Loads the prepared training split, builds the Surprise trainset, trains the
three models (Baseline, item-based KNN, SVD), and saves them.

Outputs (in ./artifacts/models/):
    trainset.pkl, baseline.pkl, knn.pkl, svd.pkl

Run:
    python 02_train.py   (run 01_prepare_data.py first)
"""
import os
import joblib
import pandas as pd
from surprise import BaselineOnly, Dataset, KNNBasic, Reader, SVD

# ============================ CONFIG - edit these ============================
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
SEED          = int(os.environ.get("SEED", 42))
RATING_SCALE  = (1, 5)

# Baseline (bias only)
BASELINE_METHOD   = "als"
BASELINE_EPOCHS   = 10
# Item-based KNN
KNN_K             = 40
KNN_SIMILARITY    = "cosine"
KNN_USER_BASED    = False          # item-based CF (more reliable on this data)
# SVD (matrix factorisation)
SVD_FACTORS       = 50
SVD_EPOCHS        = 20
SVD_LR            = 0.005
SVD_REG           = 0.02
# =============================================================================


def build_trainset(train_df):
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    return data.build_full_trainset()


def main():
    models_dir = os.path.join(ARTIFACTS_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"))
    print(f"[2/4] Building trainset from {len(train_df):,} training ratings ...")
    trainset = build_trainset(train_df)
    print(f"  Surprise trainset: {trainset.n_users:,} users, "
          f"{trainset.n_items:,} movies, {trainset.n_ratings:,} ratings")
    joblib.dump(trainset, os.path.join(models_dir, "trainset.pkl"))

    print("  Training Baseline (bias only) ...")
    baseline = BaselineOnly(
        bsl_options={"method": BASELINE_METHOD, "n_epochs": BASELINE_EPOCHS}, verbose=False
    )
    baseline.fit(trainset)
    joblib.dump(baseline, os.path.join(models_dir, "baseline.pkl"))

    print("  Training Item-based KNN ...")
    knn = KNNBasic(
        k=KNN_K, sim_options={"name": KNN_SIMILARITY, "user_based": KNN_USER_BASED}, verbose=False
    )
    knn.fit(trainset)
    joblib.dump(knn, os.path.join(models_dir, "knn.pkl"))

    print("  Training SVD ...")
    svd = SVD(n_factors=SVD_FACTORS, n_epochs=SVD_EPOCHS, lr_all=SVD_LR,
              reg_all=SVD_REG, random_state=SEED)
    svd.fit(trainset)
    joblib.dump(svd, os.path.join(models_dir, "svd.pkl"))

    print(f"[2/4] Done. Models saved to {models_dir}/")


if __name__ == "__main__":
    main()
