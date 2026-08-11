import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score



st.set_page_config(
    page_title="Housing Price Estimator",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #eeeeee;
    margin-bottom: 15px;
}

.prediction {
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title">🏠 Housing Price Estimator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Week 3 — Supervised Regression'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    This application predicts housing prices using three supervised
    regression models:
    
    **Linear Regression, Ridge Regression, and Lasso Regression.**
    """
)

@st.cache_data
def load_data():

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
            5, 5, 4, 4, 3, 3, 2, 2, 2, 2,
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


df = load_data()


features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_city_km",
    "parking_spaces"
]

X = df[features]
y = df["price"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=1.0
    ),

    "Lasso Regression": Lasso(
        alpha=0.1,
        max_iter=10000
    )
}


results = {}
coefficients = {}

for name, model in models.items():

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }

    coefficients[name] = model.coef_



metrics_df = pd.DataFrame(results).T

metrics_df = metrics_df.reset_index()

metrics_df = metrics_df.rename(
    columns={"index": "Model"}
)



best_model_name = metrics_df.loc[
    metrics_df["R2 Score"].idxmax(),
    "Model"
]

best_r2 = metrics_df.loc[
    metrics_df["R2 Score"].idxmax(),
    "R2 Score"
]



st.sidebar.header("⚙️ Model Settings")

selected_model = st.sidebar.selectbox(
    "Choose Regression Model",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.write(
    "### Models Included"
)

st.sidebar.write("• Linear Regression")
st.sidebar.write("• Ridge Regression")
st.sidebar.write("• Lasso Regression")


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Training Samples",
        len(X_train)
    )

with col2:

    st.metric(
        "Testing Samples",
        len(X_test)
    )

with col3:

    st.metric(
        "Best R² Score",
        f"{best_r2:.3f}"
    )


st.markdown("---")

st.header("🏡 Predict Housing Price")

st.write(
    "Enter the property information below."
)


col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=1500,
        step=50
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )


with col2:

    age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=10
    )

    distance = st.number_input(
        "Distance from City Center (km)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=0.5
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=1
    )


if st.button(
    "💰 Estimate Housing Price",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            age,
            distance,
            parking
        ]],
        columns=features
    )

    # Scale user input
    input_scaled = scaler.transform(
        input_data
    )

    # Select model
    selected_model_object = models[
        selected_model
    ]

    # Predict
    prediction = selected_model_object.predict(
        input_scaled
    )[0]

    st.success(
        f"Estimated Housing Price: ${prediction:,.0f}"
    )

    st.info(
        f"Prediction generated using **{selected_model}**."
    )




st.markdown("---")

st.header("📊 Regression Model Performance")

st.dataframe(
    metrics_df.style.format({
        "MSE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True
)


st.success(
    f"🏆 Best performing model: **{best_model_name}** "
    f"with an R² Score of **{best_r2:.4f}**."
)


st.subheader("R² Score Comparison")

fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)

ax1.bar(
    metrics_df["Model"],
    metrics_df["R2 Score"]
)

ax1.set_ylabel("R² Score")
ax1.set_xlabel("Regression Model")
ax1.set_title("R² Score Comparison")

ax1.set_ylim(
    min(0, metrics_df["R2 Score"].min() - 0.1),
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

st.pyplot(fig1)

plt.close(fig1)



st.subheader("Mean Squared Error (MSE) Comparison")

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

ax2.bar(
    metrics_df["Model"],
    metrics_df["MSE"]
)

ax2.set_ylabel("MSE")
ax2.set_xlabel("Regression Model")
ax2.set_title("Mean Squared Error Comparison")

plt.xticks(
    rotation=15
)

plt.tight_layout()

st.pyplot(fig2)

plt.close(fig2)



st.subheader("📈 Regression Coefficient Comparison")

coef_df = pd.DataFrame(
    coefficients,
    index=features
)

fig3, ax3 = plt.subplots(
    figsize=(11, 6)
)

coef_df.plot(
    kind="bar",
    ax=ax3
)

ax3.set_title(
    "Regression Model Coefficients"
)

ax3.set_xlabel(
    "Features"
)

ax3.set_ylabel(
    "Coefficient"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig3)

plt.close(fig3)



st.markdown("---")

st.header("📋 Housing Dataset")

st.write(
    "Dataset used to train and evaluate the regression models."
)

st.dataframe(
    df,
    use_container_width=True
)

st.markdown("---")

st.header("🎓 Week 3 Learning Outcomes")

st.write(
    """
    **Linear Regression:** Predicts a continuous target using a linear relationship
    between features and the target.

    **Ridge Regression:** Uses L2 regularization to reduce the impact of large
    coefficients and help prevent overfitting.

    **Lasso Regression:** Uses L1 regularization and can reduce some coefficients
    toward zero.

    **Mean Squared Error (MSE):** Measures the average squared difference between
    actual and predicted values. Lower values are better.

    **R² Score:** Measures how much of the variation in housing prices is explained
    by the model. Higher values are generally better.
    """
)


st.markdown("---")

st.caption(
    "Machine Learning — Week 3 | Housing Price Estimator | "
    "Supervised Regression"
)
