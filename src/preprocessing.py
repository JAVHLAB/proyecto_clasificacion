from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

import numpy as np

def preprocess_data(df):
    df = df.copy() #copia para no modificar el original

    #separar columnas numericas y categoricas
    num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "YearsExperience"]
    cat_cols = ["Gender", "Education", "EmploymentType", "City"]

    #imputacion de valores numericos por knn
    knn_imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = knn_imputer.fit_transform(df[num_cols])

    #imputacion de valores categoricos por moda (most_frequent)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    #hacer variables categoricas a numericas por one-hot encoding y ordenar las columnas
    obj_var = df["LoanApproved"] 
    dummies = pd.get_dummies(df[cat_cols])
    numerical = df[num_cols]
    df = pd.concat([numerical, dummies, obj_var], axis=1)

    #convertir valore de año a enteros
    df["Age"] = np.floor(df["Age"]).astype(int)
    df["YearsExperience"] = np.floor(df["YearsExperience"]).astype(int)

    return df