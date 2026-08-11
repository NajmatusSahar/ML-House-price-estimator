import pandas as pd
import streamlit as st

from src.feature_engineering import add_features
from src.scaler import build_preprocessor

st.set_page_config(
    page_title="Feature Scaling Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Feature Scaling Engine")
st.write(
    "A Week 2 preprocessing demo covering One-Hot Encoding, "
    "StandardScaler, MinMaxScaler, and feature engineering."
)

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sample_data.csv")
    st.info("Using the included sample dataset.")

st.subheader("Input Dataset")
st.dataframe(df, use_container_width=True)

target_column = st.selectbox(
    "Select target column",
    options=list(df.columns),
    index=list(df.columns).index("Purchased") if "Purchased" in df.columns else len(df.columns) - 1
)

scaler_type = st.radio(
    "Choose numerical scaling method",
    ["standard", "minmax"],
    horizontal=True
)

if st.button("Run Preprocessing", type="primary"):
    working_df = add_features(df)

    y = working_df[target_column]
    X = working_df.drop(columns=[target_column])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scaler_type=scaler_type
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=X.index
    )

    st.subheader("Processed Model-Ready Matrix")
    st.dataframe(processed_df, use_container_width=True)

    st.success(
        f"Preprocessing complete! Original features: {X.shape[1]} | "
        f"Processed features: {X_processed.shape[1]}"
    )

    st.subheader("Target")
    st.dataframe(y.to_frame(), use_container_width=True)
