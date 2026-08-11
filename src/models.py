from sklearn.linear_model import LinearRegression, Ridge, Lasso


def create_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    }
