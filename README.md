# 🎬 Netflix Prize — Personalized Movie Recommendation System

A reproducible movie recommendation engine built on the **Netflix Prize dataset**
(~100M ratings, 480K users, 17,770 movies). It learns user preferences, predicts
unseen ratings, and generates **explainable Top-N recommendations**.

The project compares four approaches — a bias **Baseline**, **item-based KNN**,
**matrix-factorization SVD**, and a **title-content hybrid re-ranker** — and adds an
explanation layer, a cold-start strategy, and an interactive dashboard.

> Submission for *Recommendation Systems for Personalized Content Discovery*.

---

## 📊 Results

Evaluated with a **leakage-free per-user time split**; a movie counts as relevant
for MAP@10 when its true rating is **≥ 3.5**.

| Model | RMSE ↓ | MAP@10 ↑ | Notes |
|---|:---:|:---:|---|
| Baseline (bias only) | 0.9078 | 0.0347 | Reference point; not personalized |
| Item-based KNN | 0.9653 | 0.0321 | Most explainable; powers "because you liked…" |
| **SVD** | **0.8505** | 0.0387 | Best rating accuracy; the main ranker |
| **Hybrid (re-ranked SVD)** | 0.8505 | **0.0610** | Same RMSE, **+57.6%** ranking gain |

*SVD catalogue coverage:* **26.1%** of the catalogue reached across users' Top-10
lists — healthy diversity, not the same blockbusters for everyone.

> Re-run the notebook (or the scripts) on your machine to reproduce these numbers.

---

## 📁 Repository contents

| File | Role |
|---|---|
| `01_prepare_data.py` | **Data processing pipeline** — parse raw files → subset → leakage-free time split |
| `02_train.py` | **Model training pipeline** — train Baseline, item-KNN, and SVD |
| `03_evaluate.py` | **Evaluation** — RMSE, MAP@10, hybrid re-ranker, catalogue coverage |
| `04_recommend.py` | **Recommendation generation** — Top-N lists + "because you liked…" explanations |
| `netflix-recommender.ipynb` | Full end-to-end notebook (EDA → models → evaluation → recommendations → extensions) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

**Maps to the required components:** Data Processing Pipeline → `01_prepare_data.py` ·
Model Training Pipeline → `02_train.py` · Evaluation Scripts → `03_evaluate.py` ·
Recommendation Generation Module → `04_recommend.py` · Documentation → this README +
the notebook · Reproduce instructions → below.

---

## 🚀 Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

> **Colab note:** `scikit-surprise` has a small compiled part that must match the
> installed NumPy. If `import surprise` fails with
> `numpy.core.multiarray failed to import`, run
> `pip install -q --upgrade --force-reinstall --no-cache-dir scikit-surprise`,
> then **Runtime ▸ Restart session** and re-run.

---

## 📥 Get the data

The raw dataset is **not** included (it is ~2 GB and governed by Kaggle's terms).

1. Download it from Kaggle:
   <https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data>
2. Unzip so you have these files in one folder:
   ```
   combined_data_1.txt  combined_data_2.txt
   combined_data_3.txt  combined_data_4.txt  movie_titles.csv
   ```
3. Point the code at that folder by setting **`DATA_DIR`** — in the notebook's config
   cell, and at the top of each `0X_*.py` script. On Kaggle this is usually
   `/kaggle/input/netflix-prize-data`.

---

## 🔁 Reproduce the results

### Option 1 — Notebook (simplest, runs everything)

Open **`netflix-recommender.ipynb`** in Jupyter or Google Colab, set `DATA_DIR` in the
setup cell, and **Run All**. It walks the full project top to bottom: EDA, the three
models, evaluation (RMSE + MAP@10), Top-10 recommendations, the explanation layer,
cold-start, and the hybrid re-ranker.

### Option 2 — Scripts (modular pipeline)

Run the four numbered scripts in order — each step reads the previous step's output:

```bash
python 01_prepare_data.py     # raw files → train/test split + EDA summary
python 02_train.py            # → trained Baseline / item-KNN / SVD models
python 03_evaluate.py         # → RMSE, MAP@10, hybrid, coverage
python 04_recommend.py        # → Top-N recommendations + explanations for a user
```

Intermediate outputs (the split, trained models, metrics) are written to an
`artifacts/` folder created on the first run.

---

## 🧠 Methodology (short)

- **Sparsity drives the design.** The full user×movie matrix is ~99% empty, so **SVD**
  (which generalizes across the blanks via latent features) is the main model, while
  neighbourhood CF uses **item-based** similarity (movies have far more ratings than
  users, so item–item is the more reliable axis).
- **Time-based split.** Ratings drift upward over the years, so we hold out each user's
  most-recent ratings — the honest way to test a recommender (no leakage).
- **Two metrics, on purpose.** RMSE measures rating accuracy; MAP@10 measures whether
  good movies land at the top. They reward different things, so we track both and
  prioritize ranking.
- **Hybrid re-ranking.** Title TF-IDF similarity nudges SVD's *ordering* (RMSE unchanged
  by design) and can rank brand-new, never-rated movies from their title alone.

---

## ✅ Reproducibility

- A fixed `seed` (42) threads through subsetting, the time split, SVD initialization,
  and MAP@10 user sampling.
- The four scripts run in sequence and persist their outputs, so steps are independent
  and re-runnable.
- **Reference environment:** Python 3.12 · scikit-surprise 1.1.5 · scikit-learn ·
  pandas · numpy (see `requirements.txt`).

---

## 🛣️ Roadmap

- `SVD++` / time-aware SVD to use implicit feedback and model rating drift.
- Neural collaborative filtering, with SVD as a fast first-stage candidate generator.
- Ranking-first training (BPR) to optimize Top-K directly.
- Hyper-parameter search (`GridSearchCV`) on a larger data slice.

---

## 📄 License & acknowledgements

Code released under the **MIT License**. The **Netflix Prize dataset** is © Netflix and
distributed via Kaggle under its own terms; it is **not** redistributed here. Models use
the [`scikit-surprise`](https://surpriselib.com/) library.
