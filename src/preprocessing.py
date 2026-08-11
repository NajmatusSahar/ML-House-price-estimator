import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_city_km",
    "parking_spaces",
]

TARGET = "price"


def load_data(path="data/housing.csv"):
    return pd.read_csv(path)


def prepare_data(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
