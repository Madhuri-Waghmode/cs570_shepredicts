# CS570 Project: ShePredicts

**Team**: Saya, Madhuri, Naveena

## Project Overview

The project predicts whether a user will give a movie a high rating (4 or 5 stars): `label = 1` when `Rating >= 4`,
otherwise `0`. Each rating row is one observed rating by one user for one movie. A user-movie pair missing from the
ratings data means there is no recorded rating; it is not a low rating.

- **Deliverable 1 (D1)** inspected and validated the data, defined the target, explored predictors, and planned D2. It
  does not train a model.
- **Deliverable 2 (D2)** builds the training and test modeling datasets (history features, fallback values, encoded
  feature vector) and measures two baselines: the training majority class and the rule `movie_avg >= threshold`.
  Results and explanations are in [`reports/D2.md`](reports/D2.md).

Training rows are ratings before the fixed cutoff 2000-12-01 00:00 UTC (791,270 rows); test rows are ratings from the
cutoff to 2000-12-31 (113,487 rows).

## Repository Contents

- `src/` - the D2 pipeline:
  - `config.py` - paths, the fixed cutoff, the seed, and the Spark session
  - `data.py` - reads, repairs, and joins the three files, adds the label, and splits by time
  - `features.py` - history features, fallback values, genre and gender columns, encoding, and the feature vector
  - `evaluate.py` - rule threshold choice and the baseline accuracy breakdown
  - `check_history.py`, `build_dataset.py`, `baseline.py` - the three pipeline scripts
  - `cs570_helpers.py` - shared course helpers
- `reports/` - `D2.md` and the five CSV files written by the scripts (committed exactly as written)
- `notebook/D1_deliverables.ipynb` - D1 data checks, exploratory analysis, and the plan for D2
- `CS570_L06_Lab.ipynb` - supporting Spark lab notebook

## Data

The three course-provided development files are **not** stored in Git. Copy them into `data/movie/dev/`:

- `ratings_dev.dat`
- `users_dev.dat`
- `movies_dev.dat` (Latin-1 encoded; `data.py` reads it as `ISO-8859-1`)

The files are headerless and use `::` as the field separator. `data/` is listed in `.gitignore`, and
`git ls-files data/` must print nothing. Ratings from 2001-01-01 onward are held back by the instructor and are not used.
Retain the course data's attribution and follow course rules when redistributing it.

## Requirements

The D2 pipeline was verified with:

- Python 3.14.0
- Java 17
- PySpark 4.2.0
- pandas 3.0.3
- NumPy and Matplotlib (used by `cs570_helpers.py`)

Java must be installed and available to Spark. On Windows, set `JAVA_HOME` if Java is not already discoverable.
PySpark may print a warning that pandas 3 is not fully supported; the pipeline runs and its outputs are unaffected.

## Setup

From the repository root, create and activate a virtual environment and install the dependencies.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyspark==4.2.0 pandas numpy matplotlib jupyter ipykernel
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pyspark==4.2.0 pandas numpy matplotlib jupyter ipykernel
```

## Running the Pipeline

Run the three scripts from the repository root, in this order:

```bash
python src/check_history.py
python src/build_dataset.py
python src/baseline.py
```

| Script | Checks / prints | Writes to `reports/` |
| :--- | :--- | :--- |
| `check_history.py` | History uses only strictly earlier training ratings; test history is independent of test outcomes | `history_check.csv` |
| `build_dataset.py` | Row counts, `Check, rows unchanged: True`, `Check, no nulls in the vector inputs: True` | `dataset_summary.csv`, `dataset_sample.csv`, `vector_positions.csv` |
| `baseline.py` | Majority class and chosen rule threshold | `baseline_accuracy.csv` |

All three scripts must finish without errors, and both history checks must pass.

A notebook in `notebook/` can import the pipeline modules with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
```

## Workflow

Work is done on feature branches and enters `main` only through pull requests approved by a teammate. Only `main` is
graded.
