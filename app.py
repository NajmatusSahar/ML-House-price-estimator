import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


st.set_page_config(
    page_title="AI Housing Price Estimator",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #374151 100%);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.7rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.88;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .prediction-box {
        padding: 1.4rem;
        border-radius: 15px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 3px 14px rgba(0,0,0,0.05);
    }

    .prediction-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    data = {
        "area_sqft": [
            850, 950, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
            1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
            1000, 1250, 1450, 1550, 1750, 1950, 2150, 2350, 2550, 2750
        ],
        "bedrooms": [
            2, 2, 2, 3, 3, 3, 3, 3, 3, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
            2, 3, 3, 3, 4, 4, 4, 4, 5, 5
        ],
        "bathrooms": [
            1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
            3, 3, 3, 3, 4, 4, 4, 4, 4, 5,
            1, 2, 2, 3, 3, 3, 4, 4, 4, 5
        ],
        "age_years": [
            25, 20, 15, 18, 12, 10, 8, 7, 5, 10,
            8, 6, 5, 4, 5, 3, 2, 1, 3, 2,
            30, 22, 16, 14, 12, 9, 7, 5, 4, 2
        ],
        "distance_city_km": [
            18, 15, 12, 10, 9, 8, 7, 6, 5, 6,
            5, 5, 4, 4, 3, 3, 2, 2, 1, 2,
            20, 13, 11, 9, 7, 6, 4, 3, 2, 1
        ],
        "parking_spaces": [
            0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
            0, 1, 1, 2, 2, 2, 3, 3, 3, 4
        ],
        "price": [
            95000, 115000, 135000, 150000, 170000,
            185000, 205000, 230000, 250000, 265000,
            285000, 305000, 325000, 345000, 375000,
            400000, 430000, 460000, 490000, 520000,
            90000, 145000, 175000, 215000, 255000,
            295000, 340000, 385000, 445000, 535000
        ]
    }
    return pd.DataFrame(data)


df = load_dataset()

FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_city_km",
    "parking_spaces"
]

TARGET = "price"

@st.cache_resource
def train_models(test_size, random_state, ridge_alpha, lasso_alpha):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=ridge_alpha),
        "Lasso Regression": Lasso(alpha=lasso_alpha, max_iter=10000)
    }

    results = []
    predictions = {}
    coefficients = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2
        })

        predictions[name] = pred
        coefficients[name] = model.coef_

    metrics = pd.DataFrame(results)

    return (
        models,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        coefficients,
        metrics
    )


st.sidebar.title("⚙ Model Controls")

test_percentage = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=999,
    value=42,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Regularization")

ridge_alpha = st.sidebar.slider(
    "Ridge α",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)

lasso_alpha = st.sidebar.slider(
    "Lasso α",
    min_value=0.01,
    max_value=2.0,
    value=0.1,
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Adjust the settings and the models will be retrained automatically."
)



(
    models,
    scaler,
    X_train,
    X_test,
    y_train,
    y_test,
    predictions,
    coefficients,
    metrics
) = train_models(
    test_percentage / 100,
    int(random_state),
    ridge_alpha,
    lasso_alpha
)

best_row = metrics.loc[metrics["R² Score"].idxmax()]
best_model_name = best_row["Model"]
best_r2 = best_row["R² Score"]



st.markdown("""
<div class="hero">
    <h1>■ AI Housing Price Estimator</h1>
    <p>
        Week 3 Machine Learning Project • Supervised Regression •
        Linear, Ridge & Lasso Regression
    </p>
</div>
""", unsafe_allow_html=True)



st.markdown(
    '<div class="section-title">• Model Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Dataset Rows", len(df))
c2.metric("Features", len(FEATURES))
c3.metric("Training Rows", len(X_train))
c4.metric("Testing Rows", len(X_test))
c5.metric("Best R²", f"{best_r2:.3f}")



tab_predict, tab_performance, tab_coefficients, tab_data, tab_about = st.tabs(
    [
        "■ Price Prediction",
        "▣ Model Performance",
        "▲ Coefficients",
        "▤ Dataset",
        "◆ Project Details"
    ]
)


with tab_predict:

    st.markdown(
        '<div class="section-title">■ Estimate a Property Price</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter realistic property details. The values are standardized "
        "before being passed to the selected regression model."
    )

    left, right = st.columns(2)

    with left:
        area = st.number_input(
            "• Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=1500,
            step=50
        )

        bedrooms = st.number_input(
            "• Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "• Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

    with right:
        age = st.number_input(
            "• Property Age (years)",
            min_value=0,
            max_value=100,
            value=10
        )

        distance = st.number_input(
            "• Distance from City Center (km)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.5
        )

        parking = st.number_input(
            "• Parking Spaces",
            min_value=0,
            max_value=10,
            value=1
        )

    st.markdown("---")

    selected_model = st.selectbox(
        "◆ Select Prediction Model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )

    input_df = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            age,
            distance,
            parking
        ]],
        columns=FEATURES
    )

    if st.button(
        "◆ Estimate Housing Price",
        type="primary",
        use_container_width=True
    ):

        input_scaled = scaler.transform(input_df)

        model = models[selected_model]

        prediction = model.predict(input_scaled)[0]

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">
                    Estimated Housing Price
                </div>
                <div class="prediction-value">
                    ${prediction:,.0f}
                </div>
                <div class="small-note">
                    Generated using {selected_model}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### • Input Summary")

        display_input = input_df.T.reset_index()
        display_input.columns = ["Feature", "Value"]

        st.dataframe(
            display_input,
            use_container_width=True,
            hide_index=True
        )

        prediction_file = input_df.copy()
        prediction_file["Selected Model"] = selected_model
        prediction_file["Predicted Price"] = prediction

        st.download_button(
            "↓ Download Prediction",
            data=prediction_file.to_csv(index=False),
            file_name="housing_prediction.csv",
            mime="text/csv"
        )



with tab_performance:

    st.markdown(
        '<div class="section-title">▣ Regression Performance</div>',
        unsafe_allow_html=True
    )

    formatted_metrics = metrics.copy()

    st.dataframe(
        formatted_metrics.style.format({
            "MSE": "{:,.2f}",
            "RMSE": "${:,.2f}",
            "MAE": "${:,.2f}",
            "R² Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"★ Best model based on R² Score: **{best_model_name}** "
        f"with R² = **{best_r2:.4f}**"
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader("R² Score Comparison")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            metrics["Model"],
            metrics["R² Score"]
        )

        ax.set_ylabel("R² Score")
        ax.set_ylim(
            min(0, metrics["R² Score"].min() - 0.1),
            1
        )
        ax.set_title("Higher is Better")

        plt.xticks(rotation=15)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with col_b:

        st.subheader("RMSE Comparison")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            metrics["Model"],
            metrics["RMSE"]
        )

        ax.set_ylabel("RMSE ($)")
        ax.set_title("Lower is Better")

        plt.xticks(rotation=15)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("◎ Actual vs Predicted Prices")

    actual_pred_df = pd.DataFrame({
        "Actual": y_test.values
    })

    for model_name in models:
        actual_pred_df[model_name] = predictions[model_name]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_test,
        predictions[best_model_name]
    )

    minimum = min(
        y_test.min(),
        predictions[best_model_name].min()
    )

    maximum = max(
        y_test.max(),
        predictions[best_model_name].max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(
        f"Actual vs Predicted — {best_model_name}"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("▼ Prediction Residuals")

    residuals = (
        y_test.values -
        predictions[best_model_name]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        predictions[best_model_name],
        residuals
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residual Analysis")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "↓ Download Model Metrics",
        data=metrics.to_csv(index=False),
        file_name="regression_metrics.csv",
        mime="text/csv"
    )


with tab_coefficients:

    st.markdown(
        '<div class="section-title">▲ Feature Coefficients</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Because the features are standardized, the coefficients can be "
        "compared to understand their relative influence on the prediction."
    )

    coefficient_df = pd.DataFrame(
        coefficients,
        index=FEATURES
    )

    coefficient_df["Average Absolute Impact"] = (
        coefficient_df.abs().mean(axis=1)
    )

    st.dataframe(
        coefficient_df.style.format("{:,.2f}"),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    coefficient_df.drop(
        columns=["Average Absolute Impact"]
    ).plot(
        kind="bar",
        ax=ax
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_title(
        "Regression Coefficient Comparison"
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Standardized Coefficient")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "↓ Download Coefficients",
        data=coefficient_df.to_csv(),
        file_name="model_coefficients.csv",
        mime="text/csv"
    )


# ============================================================
# DATA TAB
# ============================================================

with tab_data:

    st.markdown(
        '<div class="section-title">▤ Housing Dataset</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    a.metric(
        "Average Price",
        f"${df['price'].mean():,.0f}"
    )

    b.metric(
        "Minimum Price",
        f"${df['price'].min():,.0f}"
    )

    c.metric(
        "Maximum Price",
        f"${df['price'].max():,.0f}"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "↓ Download Dataset",
        data=df.to_csv(index=False),
        file_name="housing_dataset.csv",
        mime="text/csv"
    )

    st.subheader("• Feature Description")

    feature_info = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Property size in square feet",
            "Number of bedrooms",
            "Number of bathrooms",
            "Age of the property in years",
            "Distance from city center in kilometers",
            "Number of available parking spaces"
        ]
    })

    st.table(feature_info)


# ============================================================
# PROJECT DETAILS TAB
# ============================================================

with tab_about:

    st.markdown(
        '<div class="section-title">◆ Week 3 Learning Outcomes</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 1. Linear Regression
    Estimates a continuous target by learning a linear relationship
    between input features and housing price.

    ### 2. Ridge Regression
    Adds L2 regularization to penalize large coefficients and reduce
    overfitting.

    ### 3. Lasso Regression
    Uses L1 regularization and can shrink less important coefficients
    toward zero.

    ### 4. Mean Squared Error (MSE)
    Measures the average squared difference between actual and predicted
    prices. Lower MSE indicates better prediction performance.

    ### 5. R² Score
    Measures the proportion of variance in housing prices explained by
    the regression model. Higher values are generally better.

    ### 6. Feature Scaling
    StandardScaler transforms numerical features to a common scale
    before model training.

    ### 7. Model Evaluation
    Models are evaluated using MSE, RMSE, MAE and R² Score.
    """)

    st.markdown("---")

    st.subheader("→ Machine Learning Pipeline")

    st.code("""
Raw Housing Data
       ↓
Feature Selection
       ↓
Train / Test Split
       ↓
StandardScaler
       ↓
Linear Regression
Ridge Regression
Lasso Regression
       ↓
Predictions
       ↓
MSE / RMSE / MAE / R²
       ↓
Model Comparison
       ↓
Housing Price Prediction
    """, language="text")

    st.markdown("---")

    st.subheader("▤ Recommended GitHub Structure")

    st.code("""
ml-house-price-estimator/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── housing.csv
│
├── models/
│   └── (optional saved models)
│
├── reports/
│   └── (optional generated reports)
│
└── src/
    └── (optional supporting modules)
    """, language="text")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Learning Week 3 • Housing Price Estimator • "
    "Supervised Regression"
)
