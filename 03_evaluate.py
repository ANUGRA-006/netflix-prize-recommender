#!/usr/bin/env python
"""Step 3/4 — Evaluation.

Computes RMSE (all models) and MAP@10 (all models + the hybrid re-ranker),
plus catalogue coverage, then writes metrics.json and a comparison chart.

Outputs (in artifacts/):
    metrics.json, model_comparison.png

Run:
    python scripts/03_evaluate.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.config import load_config
from recsys.evaluate import (
    average_precision_at_k,
    build_test_index,
    evaluable_users,
    map_at_k,
    rmse,
    sample_eval_users,
)
from recsys.hybrid import build_content_model, hybrid_top_n
from recsys.recommend import all_item_ids, build_user_history, catalogue_coverage


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    a = cfg.artifacts_dir

    train_df = pd.read_parquet(a / "train.parquet")
    test_df = pd.read_parquet(a / "test.parquet")
    movies = pd.read_parquet(a / "movies.parquet")

    trainset = joblib.load(cfg.models_dir / "trainset.pkl")
    models = {
        "Baseline": joblib.load(cfg.models_dir / "baseline.pkl"),
        "Item-based KNN": joblib.load(cfg.models_dir / "knn.pkl"),
        "SVD": joblib.load(cfg.models_dir / "svd.pkl"),
    }
    svd = models["SVD"]

    all_items = all_item_ids(trainset)
    history = build_user_history(train_df)
    testset = list(test_df[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None))
    test_by_user = build_test_index(test_df)
    thr, k = cfg.eval.relevance_threshold, cfg.eval.k

    # ---- RMSE -------------------------------------------------------------
    print("[3/4] RMSE ...")
    results = {}
    for name, model in models.items():
        results[name] = {"RMSE": round(rmse(model, testset), 4)}
        print(f"  {name:>16}  RMSE = {results[name]['RMSE']:.4f}")

    # ---- MAP@K ------------------------------------------------------------
    users = evaluable_users(test_by_user, thr)
    eval_sample = sample_eval_users(users, cfg.eval.max_eval_users, rng)
    print(f"  Evaluating MAP@{k} on {len(eval_sample):,} users "
          f"(of {len(users):,} evaluable) ...")
    for name, model in models.items():
        t0 = time.time()
        results[name][f"MAP@{k}"] = round(
            map_at_k(model, eval_sample, all_items=all_items, history=history,
                     test_by_user=test_by_user, threshold=thr, k=k), 4
        )
        print(f"  {name:>16}  MAP@{k} = {results[name][f'MAP@{k}']:.4f}  "
              f"({time.time() - t0:.1f}s)")

    # ---- Hybrid re-ranker (RMSE == SVD by design) -------------------------
    print("  Building hybrid content model + scoring ...")
    content = build_content_model(
        movies, all_items, min_df=cfg.hybrid.tfidf_min_df,
        ngram_max=cfg.hybrid.tfidf_ngram_max,
    )
    hyb_aps = []
    for u in eval_sample:
        top = hybrid_top_n(u, svd=svd, content=content, train_df=train_df,
                           history=history, n=k, w=cfg.hybrid.svd_weight)
        relevant = {m for m, r in test_by_user[u] if r >= thr}
        hyb_aps.append(average_precision_at_k(top, relevant, k=k))
    results["Hybrid (re-ranked SVD)"] = {
        "RMSE": results["SVD"]["RMSE"],
        f"MAP@{k}": round(float(np.mean(hyb_aps)), 4),
    }
    print(f"  {'Hybrid':>16}  MAP@{k} = {results['Hybrid (re-ranked SVD)'][f'MAP@{k}']:.4f}")

    # ---- Coverage ---------------------------------------------------------
    cov = catalogue_coverage(svd, eval_sample, all_items, history, k=k)
    print(f"  SVD catalogue coverage: {cov:.1%} of {len(all_items):,} movies")

    out = {
        "metrics": results,
        "n_eval_users": int(len(eval_sample)),
        "catalogue_coverage_svd": round(cov, 4),
        "relevance_threshold": thr,
        "k": k,
    }
    with open(a / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    _save_chart(results, k, a / "model_comparison.png")
    print(f"[3/4] Done. metrics.json + model_comparison.png in {a}/")


def _save_chart(results, k, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(results.keys())
        rmse_v = [results[n]["RMSE"] for n in names]
        map_v = [results[n][f"MAP@{k}"] for n in names]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].barh(names, rmse_v, color="#3b528b"); ax[0].invert_yaxis()
        ax[0].set_title("RMSE (lower is better)")
        ax[1].barh(names, map_v, color="#21918c"); ax[1].invert_yaxis()
        ax[1].set_title(f"MAP@{k} (higher is better)")
        plt.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # chart is a nicety, never fail the run for it
        print(f"  (chart skipped: {e})")


if __name__ == "__main__":
    main()
