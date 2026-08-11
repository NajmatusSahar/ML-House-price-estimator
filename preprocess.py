import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import add_features
from src.scaler import build_preprocessor


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    target = df["Purchased"]
    X = df.drop(columns=["Purchased"])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scaler_type="standard"
    )

    X_processed = preprocessor.fit_transform(X)

    print("Original shape:", X.shape)
    print("Processed shape:", X_processed.shape)
    print("\nProcessed feature matrix:")
    print(X_processed)


if __name__ == "__main__":
    main()
