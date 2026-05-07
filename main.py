from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
import joblib
import os

#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt

df = load_data("datos/raw/loan_risk_prediction_dataset.csv")
df = preprocess_data(df)

df.to_csv("datos/processed/loan_risk_processed.csv", index=False)

X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

model, X_test, y_test = train_model(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/decision_tree_model.pkl")
joblib.dump(list(X.columns), "models/columns.pkl")

acc, cm = evaluate_model(model, X_test, y_test)

print("Porcentaje de eficiencia:", acc)
print("Matriz de confusión:\n", cm)
