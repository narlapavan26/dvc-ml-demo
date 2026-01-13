import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/iris_clean.csv")
X = df.drop(columns=["target"])
y = df["target"]

model = joblib.load("models/model.pkl")
preds = model.predict(X)

with open("metrics/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy_score(y, preds)}, f)
