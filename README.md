# Week 3 — Housing Price Estimator

A complete Machine Learning Week 3 project covering:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Mean Squared Error (MSE)
- R² Score
- Feature scaling
- Regression metric comparison
- Coefficient charts
- Streamlit demo

## Project Structure

```text
ml_week3_housing_price_estimator/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── housing.csv
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── models.py
    └── evaluation.py
```

## 1. Install

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 2. Train the models

```bash
python train.py
```

This creates:
- trained models in `models/`
- regression metrics in `reports/regression_metrics.csv`
- coefficient chart in `reports/coefficient_comparison.png`

## 3. Run the Streamlit application

```bash
streamlit run app.py
```

## 4. GitHub

Create a public GitHub repository and push this folder:

```bash
git init
git add .
git commit -m "Week 3 Housing Price Estimator"
git branch -M main
git remote add origin https://github.com/NajmatusSahar/ML-House-price-estimator
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## 5. Streamlit deployment

After pushing to GitHub:
1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Select this repository.
4. Select `app.py` as the main file.
5. Deploy.

Your deployed app URL can then be submitted as the mandatory Live Demo URL.

## Dataset

The included `data/housing.csv` is a small educational housing dataset with numerical features. The target is `price`.

Features:
- area_sqft
- bedrooms
- bathrooms
- age_years
- distance_city_km
- parking_spaces

Target:
- price

## Learning Outcomes

By completing this project, you demonstrate:
- supervised regression
- feature scaling
- Linear Regression
- Ridge regularization
- Lasso regularization
- MSE evaluation
- R² evaluation
- model comparison
- coefficient interpretation
- deployment using Streamlit
