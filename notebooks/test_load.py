import pandas as pd
import os

print("CWD:", os.getcwd())

possible_paths = [
    '../data/raw/loan_data.csv',         # From notebooks dir
    'data/raw/loan_data.csv',            # From project root
    '../data/raw/Loan_default.csv',      # Alternative file
    'data/raw/Loan_default.csv'          # Alternative file from root
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            print(f"Columns: {df.columns.tolist()}")
            break
        except Exception as e:
            print(f"Found file at {path} but failed to load: {e}")

if df is None:
    print("FAILED to load data from any path.")
    exit(1)
else:
    print("SUCCESS")
