# 🎬 Netflix Prize — Personalized Movie Recommendation System

A clean, reproducible recommendation engine built on the **Netflix Prize dataset**
(~100M ratings, 480K users, 17,770 movies). It learns user preferences, predicts
unseen ratings, and generates **explainable Top-N recommendations**.

The project compares four approaches — a bias **Baseline**, **item-based KNN**,
**matrix-factorization SVD**, and a **title-content hybrid re-ranker** — and adds
an explanation layer, a cold-start strategy, and an interactive dashboard.

> Submission for *Recommendation Systems for Personalized Content Discovery*.
> See `docs/` for the technical report (Deliverable 1).

---

## 📊 Results at a glance

Evaluated with a **leakage-free per-user time split**; a movie counts as relevant
for MAP@10 when its true rating is **≥ 3.5**.

| Model | RMSE ↓ | MAP@10 ↑ | Notes |
|---|:---:|:---:|---|
| Baseline (bias only) | 0.9078 | 0.0347 | Reference point; not personalized |
| Item-based KNN | 0.9653 | 0.0321 | Most explainable; used for "because you liked…" |
| **SVD** | **0.8505** | 0.0387 | Best rating accuracy; the main ranker |
| **Hybrid (re-ranked SVD)** | 0.8505 | **0.0610** | Same RMSE, **+57.6%** ranking gain |

*SVD catalogue coverage:* **26.1%** of the catalogue reached across users' Top-10
lists (healthy diversity, not the same blockbusters for everyone).

> Numbers above are from a run on the full Netflix subset (2,000 most-rated movies
> × 10,000 sampled users). They are reproducible with the default `config.yaml`.

---

## ✨ Highlights

- **End-to-end, scripted pipeline** — `prepare → train → evaluate → recommend`,
  each step a standalone CLI that reads/writes versioned artifacts.
- **Config-driven & reproducible** — every knob lives in `config.yaml`; a fixed
  `seed` makes runs deterministic.
- **Honest evaluation** — per-user time split (no leakage), both **RMSE** and
  **MAP@10**, plus catalogue coverage.
- **Explainable** — SVD ranks, KNN explains, with truthful "because you liked…"
  reasons (and it admits when there's no strong look-alike).
- **Cold-start aware** — damped-popularity for new users; a popularity↔SVD blend
  that trusts SVD more as history grows.
- **Verifiable without the 2 GB dataset** — a synthetic-data generator + smoke
  test exercise the whole pipeline (`make test`).

---

## 🗂️ Repository structure

```
netflix-prize-recommender/
├── config.yaml               # all hyperparameters & paths (single source of truth)
├── requirements.txt          # pinned dependencies
├── pyproject.toml            # installable package (pip install -e .)
├── Makefile                  # one-command targets (make all / make test)
│
├── recsys/                   # ── the library ──────────────────────────────
│   ├── config.py             #   load config.yaml (+ env overrides)
│   ├── data.py               #   ▶ Data Processing Pipeline (parse, subset, split)
│   ├── models.py             #   ▶ Model Training Pipeline (Baseline, KNN, SVD)
│   ├── evaluate.py           #   ▶ Evaluation (RMSE, MAP@K, AP@K)
│   ├── recommend.py          #   ▶ Recommendation Generation Module (Top-N)
│   ├── explain.py            #     explanation layer (KNN similarity)
│   ├── coldstart.py          #     damped popularity + popularity↔SVD blend
│   └── hybrid.py             #     TF-IDF title content re-ranker
│
├── scripts/                  # ── runnable pipeline steps ──────────────────
│   ├── 01_prepare_data.py    #   raw files  → train/test parquet + EDA summary
│   ├── 02_train.py           #   train split → fitted models (.pkl)
│   ├── 03_evaluate.py        #   models      → metrics.json + comparison chart
│   └── 04_recommend.py       #   models      → Top-N + explanations for a user
│
├── tests/                    # ── reproducibility checks ───────────────────
│   ├── make_synthetic_data.py#   tiny fake dataset in the real file format
│   └── test_smoke.py         #   end-to-end test of every module (no dataset)
│
├── notebooks/
│   └── netflix_recommender.ipynb   # original exploratory notebook
├── data/                     # dataset goes here (see data/README.md) — gitignored
└── docs/                     # technical report (Deliverable 1)
```

**Where each required component lives:** Data Processing Pipeline → `recsys/data.py`
+ `scripts/01_prepare_data.py` · Model Training Pipeline → `recsys/models.py` +
`scripts/02_train.py` · Evaluation Scripts → `recsys/evaluate.py` +
`scripts/03_evaluate.py` · Recommendation Generation Module → `recsys/recommend.py`
+ `scripts/04_recommend.py` · Documentation → this README + module docstrings ·
Reproduce instructions → below.

---

## 🚀 Quickstart

### 1. Install

```bash
git clone <your-repo-url>
cd netflix-prize-recommender

python -m venv .venv && source .venv/bin/activate      # optional but recommended
make setup                                             # installs deps + the package
# (equivalently: pip install -r requirements.txt && pip install -e .)
```

> **Colab note:** `scikit-surprise` has a small compiled part that must match the
> installed NumPy. If `import surprise` fails with
> `numpy.core.multiarray failed to import`, run
> `pip install -q --upgrade --force-reinstall --no-cache-dir scikit-surprise`,
> then **Runtime ▸ Restart session**, and re-run. Pinning `scikit-surprise==1.1.5`
> (see `requirements.txt`) avoids this.

### 2. Get the data

Download the Netflix Prize dataset and unzip it into `data/raw/`
(see **[`data/README.md`](data/README.md)**), or set `data_dir` in `config.yaml`
to wherever the files already are (e.g. `/kaggle/input/netflix-prize-data`).

### 3. Run the full pipeline

```bash
make all          # = prepare → train → evaluate
```

or step by step:

```bash
python scripts/01_prepare_data.py     # → artifacts/train.parquet, test.parquet, ...
python scripts/02_train.py            # → artifacts/models/{baseline,knn,svd}.pkl
python scripts/03_evaluate.py         # → artifacts/metrics.json + model_comparison.png
python scripts/04_recommend.py --user 1621025 --n 10
```

### 4. Don't have the dataset? Verify it works anyway

```bash
make test         # generates synthetic data and runs the whole pipeline end-to-end
```

---

## ⚙️ Configuration

Everything is controlled by **`config.yaml`** — no need to touch the code. Lower
the subset sizes if your machine is small; raise them to use more data.

| Key | Default | Meaning |
|---|---|---|
| `seed` | 42 | global RNG seed (determinism) |
| `data_dir` | `data/raw` | folder with the raw Kaggle files |
| `subset.n_top_movies` | 2000 | keep the N most-rated movies |
| `subset.n_users` | 10000 | sample this many active users |
| `subset.min_user_ratings` | 20 | min ratings a user must have |
| `split.test_fraction` | 0.20 | each user's most-recent 20% held out |
| `eval.relevance_threshold` | 3.5 | rating ≥ this ⇒ "relevant" for MAP@10 |
| `eval.k` | 10 | Top-K for MAP@K |
| `eval.max_eval_users` | 1000 | cap users scored for ranking metrics |
| `models.svd.*` | 50f / 20ep | SVD factors, epochs, lr, reg |
| `models.knn.*` | k=40, cosine | item-based KNN neighbours & similarity |
| `coldstart.damping` | 200 | prior strength for damped popularity |
| `hybrid.svd_weight` | 0.7 | SVD vs content weight in re-ranking |

Any top-level scalar can also be overridden by an env var:
`DATA_DIR=/path SEED=7 python scripts/01_prepare_data.py`.

---

## 🧠 Methodology (short)

- **Sparsity drives the design.** The full matrix is ~99% empty, so **SVD** (which
  generalizes across the blanks via latent features) is the main model, while
  neighbourhood CF uses **item-based** similarity (movies have more ratings than
  users, so item-item is the more reliable axis).
- **Time-based split.** Ratings drift upward over the years, so we hold out each
  user's most-recent ratings — the honest way to test a recommender.
- **Two metrics, on purpose.** RMSE measures rating accuracy; MAP@10 measures
  whether good movies land at the top. They reward different things, so we track
  both and prioritize ranking.
- **Hybrid re-ranking.** Title TF-IDF similarity nudges SVD's *ordering* (RMSE
  unchanged by design), and can rank brand-new, never-rated movies from the title
  alone.

Full write-up in `docs/` (Deliverable 1 technical report).

---

## 🔁 Reproducibility

- Single `seed` (default 42) threads through subsetting, the time split, SVD
  initialization, and MAP@10 user sampling.
- Each pipeline step persists its outputs to `artifacts/`, so steps are
  independent and re-runnable.
- `metrics.json` records the exact RMSE / MAP@10 / coverage of a run.
- `make test` proves the pipeline runs end-to-end on synthetic data.

**Reference environment:** Python 3.12 · numpy 2.x · pandas 2.x ·
scikit-learn 1.6 · scikit-surprise 1.1.5 (see `requirements.txt`).

---

## 🧪 Testing

```bash
make test          # full smoke test on synthetic data (22 checks)
```

The smoke test validates parsing (incl. commas-in-titles and `NULL` years),
the no-leakage split, model training, RMSE/MAP@10 ranges, Top-N sanity
(sorted, no already-seen items), explanations, cold-start, and the hybrid.

---

## 🛣️ Roadmap

- `SVD++` / time-aware SVD to use implicit feedback and model rating drift.
- Neural collaborative filtering, with SVD as a fast first-stage candidate
  generator.
- Ranking-first training (BPR) to optimize Top-K directly.
- Hyper-parameter search (`GridSearchCV`) on a larger data slice.

---

## 📄 License & acknowledgements

Code released under the **MIT License** (see `LICENSE`). The **Netflix Prize
dataset** is © Netflix and distributed via Kaggle under its own terms; it is
**not** redistributed in this repository. Models use the excellent
[`scikit-surprise`](https://surpriselib.com/) library.
