import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    processed_df = pd.DataFrame(X_processed)
    processed_df["Churn"] = y.values

    processed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(
        "churn-bigml-80_raw/churn-bigml-80.csv",
        "preprocessing/churn-bigml-80_preprocessing/churn_processed.csv"
    )