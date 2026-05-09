import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def load_data(file_path):
    """Purpose: Read the churn CSV into a pandas DataFrame for analysis or modeling.

    Steps:
        1. Accept a filesystem path to the CSV (typically under `data/`).
        2. Parse with pandas.read_csv and return the loaded frame.
    """
    return pd.read_csv(file_path)


def analyze_data(data):
    """Purpose: Summarize numeric columns for quick exploratory understanding.

    Steps:
        1. Take a DataFrame (expected to include numeric feature columns).
        2. Call `pandas.DataFrame.describe()` for count/mean/std/min/max/quartiles.
        3. Return that summary frame for display or logging.
    """
    return data.describe()


if __name__ == "__main__":
    file_path = os.path.join(PROJECT_ROOT, "data", "customer_churn.csv")
    data = load_data(file_path)
    print(analyze_data(data))
