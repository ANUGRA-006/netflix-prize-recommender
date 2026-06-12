#!/usr/bin/env python
"""
Step 3/4 - Evaluation  (standalone)

Computes RMSE (all models) and MAP@10 (all models + the hybrid re-ranker),
plus catalogue coverage, then writes metrics.json and a comparison chart.

Mandatory metrics:
    RMSE   - rating-prediction accuracy
    MAP@10 - ranking quality; a movie is relevant when true rating >= 3.5

Outputs (in ./artifacts/):
    metrics.json, model_comparison.png

Run:
    python 03_evaluate.py   (run 01 and 02 first)
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================ CONFIG - edit these ============================
ARTIFACTS_DIR       = os.environ.get("ARTIFACTS_DIR", "artifacts")
SEED                = int(os.environ.get("SEED", 42))
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", 3.5))
K                   = int(os.environ.get("K", 10))
MAX_EVAL_USERS      = int(os.environ.get("MAX_EVAL_USERS", 1000))  # cap users for ranking metrics
HYBRID_SVD_WEIGHT   = 0.7      # final = w*SVD_rank + (1-w)*content_rank
TFIDF_MIN_DF        = 2
TFIDF_NGRAM_MAX     = 2
# =============================================================================


# ---- shared helpers ----
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


def average_precision_at_k(ranked_items, relevant, k=10):
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(k, len(relevant))


def rmse(model, testset):
    return float(accuracy.rmse(model.test(testset), verbose=False))


def map_at_k(model, users, all_items, history, test_by_user, threshold, k):
    aps = []
    for u in users:
        items, _ = recommend_top_n(model, u, all_items, history, n=k)
        relevant = {m for m, r in test_by_user[u] if r >= threshold}
        aps.append(average_precision_at_k(list(items), relevant, k=k))
    return float(np.mean(aps)) if aps else 0.0


def catalogue_coverage(model, users, all_items, history, k=10):
    reached = set()
    for u in users:
        items, _ = recommend_top_n(model, u, all_items, history, n=k)
        reached.update(items.tolist())
    return len(reached) / len(all_items)


# ---- hybrid content re-ranker ----
def build_content_sim(movies, all_items):
    item_order = list(all_items)
    titles = (movies.reindex(item_order).title.fillna("")
              .str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True))
    tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, ngram_range=(1, TFIDF_NGRAM_MAX), stop_words="english")
    vectors = tfidf.fit_transform(titles)
    sim = cosine_similarity(vectors)
    pos_of = {m: i for i, m in enumerate(item_order)}
    return item_order, sim, pos_of


def hybrid_top_n(user_id, svd, item_order, content_sim, pos_of, train_df, history, n=10, w=0.7):
    seen = history.get(user_id, set())
    liked = train_df[(train_df.user_id == user_id) & (train_df.rating >= 4)].movie_id.values
    liked_pos = [pos_of[m] for m in liked if m in pos_of]
    candidates = [m for m in item_order if m not in seen]
    cand_pos = [pos_of[m] for m in candidates]
    svd_scores = np.array([svd.predict(user_id, int(m)).est for m in candidates])
    if liked_pos:
        content_score = content_sim[np.ix_(cand_pos, liked_pos)].mean(axis=1)
        s_norm = (svd_scores - svd_scores.min()) / (np.ptp(svd_scores) + 1e-9)
        c_norm = (content_score - content_score.min()) / (np.ptp(content_score) + 1e-9)
        final = w * s_norm + (1 - w) * c_norm
    else:
        final = svd_scores
    order = np.argsort(-final)[:n]
    return [candidates[i] for i in order]


def save_chart(results, k, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(results.keys())
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].barh(names, [results[n]["RMSE"] for n in names], color="#3b528b"); ax[0].invert_yaxis()
        ax[0].set_title("RMSE (lower is better)")
        ax[1].barh(names, [results[n][f"MAP@{k}"] for n in names], color="#21918c"); ax[1].invert_yaxis()
        ax[1].set_title(f"MAP@{k} (higher is better)")
        plt.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"  (chart skipped: {e})")


def main():
    rng = np.random.default_rng(SEED)
    md = os.path.join(ARTIFACTS_DIR, "models")

    train_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"))
    test_df  = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))
    movies   = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies.parquet"))

    trainset = joblib.load(os.path.join(md, "trainset.pkl"))
    models = {
        "Baseline":       joblib.load(os.path.join(md, "baseline.pkl")),
        "Item-based KNN": joblib.load(os.path.join(md, "knn.pkl")),
        "SVD":            joblib.load(os.path.join(md, "svd.pkl")),
    }
    svd = models["SVD"]

    all_items = all_item_ids(trainset)
    history = build_user_history(train_df)
    testset = list(test_df[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None))
    test_by_user = {}
    for u, m, r in zip(test_df.user_id.values, test_df.movie_id.values, test_df.rating.values):
        test_by_user.setdefault(u, []).append((m, r))

    print("[3/4] RMSE ...")
    results = {}
    for name, model in models.items():
        results[name] = {"RMSE": round(rmse(model, testset), 4)}
        print(f"  {name:>16}  RMSE = {results[name]['RMSE']:.4f}")

    users = [u for u, lst in test_by_user.items() if any(r >= RELEVANCE_THRESHOLD for _, r in lst)]
    users = np.array(users)
    eval_sample = (rng.choice(users, size=MAX_EVAL_USERS, replace=False)
                   if len(users) > MAX_EVAL_USERS else users)
    print(f"  Evaluating MAP@{K} on {len(eval_sample):,} users (of {len(users):,} evaluable) ...")
    for name, model in models.items():
        t0 = time.time()
        results[name][f"MAP@{K}"] = round(
            map_at_k(model, eval_sample, all_items, history, test_by_user, RELEVANCE_THRESHOLD, K), 4)
        print(f"  {name:>16}  MAP@{K} = {results[name][f'MAP@{K}']:.4f}  ({time.time()-t0:.1f}s)")

    print("  Building hybrid content model + scoring ...")
    item_order, content_sim, pos_of = build_content_sim(movies, all_items)
    hyb_aps = []
    for u in eval_sample:
        top = hybrid_top_n(u, svd, item_order, content_sim, pos_of, train_df, history,
                           n=K, w=HYBRID_SVD_WEIGHT)
        relevant = {m for m, r in test_by_user[u] if r >= RELEVANCE_THRESHOLD}
        hyb_aps.append(average_precision_at_k(top, relevant, k=K))
    results["Hybrid (re-ranked SVD)"] = {
        "RMSE": results["SVD"]["RMSE"],
        f"MAP@{K}": round(float(np.mean(hyb_aps)), 4),
    }
    print(f"  {'Hybrid':>16}  MAP@{K} = {results['Hybrid (re-ranked SVD)'][f'MAP@{K}']:.4f}")

    cov = catalogue_coverage(svd, eval_sample, all_items, history, k=K)
    print(f"  SVD catalogue coverage: {cov:.1%} of {len(all_items):,} movies")

    out = {"metrics": results, "n_eval_users": int(len(eval_sample)),
           "catalogue_coverage_svd": round(cov, 4),
           "relevance_threshold": RELEVANCE_THRESHOLD, "k": K}
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_chart(results, K, os.path.join(ARTIFACTS_DIR, "model_comparison.png"))
    print(f"[3/4] Done. metrics.json + model_comparison.png in {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
