import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import load_data, prepare_data, FEATURES
from src.models import create_models
from src.evaluation import evaluate_model


os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = load_data()

(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    scaler,
) = prepare_data(df)

models = create_models()

metrics = []
coefficients = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    result = evaluate_model(model, X_test_scaled, y_test)
    result["Model"] = name
    metrics.append(result)

    coefficients[name] = model.coef_

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join("models", filename))

joblib.dump(scaler, "models/scaler.joblib")

metrics_df = pd.DataFrame(metrics)[["Model", "MSE", "RMSE", "R2 Score"]]
metrics_df.to_csv("reports/regression_metrics.csv", index=False)

coef_df = pd.DataFrame(coefficients, index=FEATURES)
coef_df.to_csv("reports/model_coefficients.csv")

ax = coef_df.plot(kind="bar", figsize=(12, 6))
ax.set_title("Regression Model Coefficients")
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/coefficient_comparison.png", dpi=150)
plt.close()

print("\nRegression Metrics:")
print(metrics_df.to_string(index=False))

best_model = metrics_df.sort_values("R2 Score", ascending=False).iloc[0]
print(
    f"\nBest model by R² Score: {best_model['Model']} "
    f"(R² = {best_model['R2 Score']:.4f})"
)
print("\nTraining completed successfully.")
