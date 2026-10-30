# CS570 Project: ShePredicts

## Project Overview

This repository contains Deliverable 1 (D1) for the ShePredicts CS570 project. The project studies whether a user will give 
a movie a high rating (4 or 5 stars). D1 inspects and validates the supplied movie, user, and rating data; defines the prediction
target; explores potential predictors; and outlines the next deliverable. **D1 is exploratory and does not train a model.**

Each rating row represents an observed rating by one user for one movie. A user-movie pair missing from the ratings data means there 
is no recorded rating; it should not be interpreted as a low rating.

## Repository Contents

- `notebook/D1_deliverables.ipynb` - D1 data checks, exploratory analysis, saved results, and plan for D2.
- `src/cs570_helpers.py` - shared PySpark analysis, metrics, and plotting helpers.
- `data/movie/dev/` - course-provided development data files:
  - `movies_dev.dat`
  - `ratings_dev.dat`
  - `users_dev.dat`
- `CS570_L06_Lab.ipynb` - supporting Spark lab notebook.

The data files are headerless and use `::` as the field separator. The D1 notebook handles the movie-title encoding when it reads `movies_dev.dat`. Retain the course data's attribution and follow course rules when redistributing it.

## Requirements

The D1 notebook was verified with:

- Python 3.12.14
- Java 17
- PySpark 3.5.3
- NumPy
- Matplotlib
- Jupyter

Java must be installed and available to Spark. On Windows, configure `JAVA_HOME` if Java is not already discoverable.

## Setup on Windows

From the repository root in PowerShell, create and activate a virtual environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pyspark==3.5.3 numpy matplotlib jupyter ipykernel
