import pandas as pd
import numpy as np

def extract_meta_features(df: pd.DataFrame):
    info = {}
    info["type"] = "Tabular Data"
    info["num_rows"] = int(df.shape[0])
    info["num_columns"] = int(df.shape[1])
    info["num_numeric_columns"] = int(df.select_dtypes(include=[np.number]).shape[1])
    info["num_categorical_columns"] = int(df.select_dtypes(exclude=[np.number]).shape[1])
    info["missing_values_total"] = int(df.isna().sum().sum()) if df.size > 0 else 0
    return info