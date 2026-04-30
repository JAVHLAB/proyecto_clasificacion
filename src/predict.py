import joblib
import pandas as pd
from src.preprocessing import preprocess_data

MODEL_PATH = "models/decision_tree_model.pkl"
COLUMNS_PATH = "models/columns.pkl"

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    df_processed = preprocess_data(df)
    target_col = "LoanApproved"
    if target_col in df_processed.columns:
        df_processed = df_processed.drop(columns=[target_col])
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df_processed)
    return prediction[0]
