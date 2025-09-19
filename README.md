# 📘 QuantChallenge 2025

Repository for our team’s participation in **QuantChallenge 2025**.
The competition has two main components:

1. **Quantitative Market Research** → predict market features with highest possible R².
2. **Live Algorithmic Trading** → build real-time trading algorithms.

This repo is structured to keep research, code, and experiments organized.

---

## 📂 Project Structure

```
quantchallenge-2025/
│
├── notebooks/       # Jupyter notebooks (EDA, experiments)
├── src/             # Reusable code modules (data, features, models, trading)
├── data/            # Raw data (not tracked in GitHub)
├── outputs/         # Model predictions, plots, logs
├── README.md        # This file
└── .gitignore
```

---

## ⚙️ Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/YOUR_USERNAME/quantchallenge-2025.git
   cd quantchallenge-2025
   ```
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Workflow

### Market Research (Sept 19–22)

* Work in `notebooks/` for experiments.
* Move reusable code into `src/` (e.g., data loading, feature engineering).
* Save submission files to `outputs/`.

### Live Trading (Sept 22–25)

* Implement trading algorithms in `src/trading/`.
* Use sandbox to test before submission.

---

## 🤝 Team Workflow

* **Branching**:

  ```bash
  git checkout -b feature-name
  git push origin feature-name
  ```

* **Pull Requests**:

  * Open a PR when ready for review.
  * At least 1 teammate must review before merging into `main`.

* **Naming convention**:

  * Notebooks: `01_baseline.ipynb`, `02_xgboost.ipynb`
  * Branches: `feature-xgboost`, `bugfix-preprocessing`
  * Commits: short, descriptive (e.g., `add ridge baseline model`)

---

## 🏆 Submission Guidelines

* Always keep the **latest valid submission** in `outputs/`.
* Record submission details in a shared `submissions_log.md`:

  ```
  Date | Model | Features | CV Score | Public LB | Notes
  -----|-------|----------|----------|-----------|------
  2025-09-19 | Ridge | raw A..N | 0.75 / 0.53 | TBD | baseline
  ```

---

## 📌 Notes

* Do **not** commit raw data (it’s ignored in `.gitignore`).
* Use `.gitkeep` files to preserve empty folders in Git.
* Always check in code & notebooks, never only outputs.
