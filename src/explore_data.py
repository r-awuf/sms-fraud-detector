from pathlib import Path

import pandas as pd

data_path = Path(__file__).resolve().parent.parent / "data" / "spam.csv"
df = pd.read_csv(data_path, encoding="latin-1")

print(df.head())
print(df.shape)
print(df.columns)
