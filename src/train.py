import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/processed/iris_clean.csv")
X = df.drop(columns=["target"])
y = df["target"]

model = LogisticRegression(max_iter=500)
model.fit(X, y)

joblib.dump(model, "models/model.pkl")
