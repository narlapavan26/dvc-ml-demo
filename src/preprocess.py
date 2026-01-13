import pandas as pd

df = pd.read_csv("data/raw/iris.csv")
df = df.sample(frac=1, random_state=42)
df.to_csv("data/processed/iris_clean.csv", index=False)
