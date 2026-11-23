# Used Bike Price Prediction Using Ensemble Learning

An ensemble learning project that compares multiple machine learning approaches for predicting used bike prices in Sri Lanka. The project includes a model evaluation pipeline and an interactive Streamlit dashboard for comparing model performance, exploring feature importance, and generating price predictions.

## 1. Project Overview

This project uses a dataset of **5,016 used-bike listings** collected from ikman.lk between January and March 2022.

The objective is to investigate how different ensemble learning techniques perform on a real-world Sri Lankan tabular regression dataset.

- **Prediction target:** Bike price (LKR)
- **Problem type:** Supervised regression
- **Dataset size:** 5,016 listings
- **Training/testing split:** 80% training, 20% testing
- **Random state:** 42

## 2. Dataset Features

The following features were selected for model training:

| Feature | Description |
|---|---|
| Bike Type | Type of bike |
| Brand | Manufacturer |
| Trim/Edition | Bike edition or trim |
| Model | Bike model |
| Year | Manufacturing year |
| Mileage | Distance travelled |
| Capacity | Engine capacity |

The target variable is `Price`.

Irrelevant and redundant columns were excluded. Missing categorical values were handled using an `Unknown` category, while missing numerical values were imputed using the median. Categorical features were encoded using One-Hot Encoding.

Numerical features were scaled for the ANN model.

## 3. Models Implemented

Four ensemble learning approaches were investigated.

### Bagging
- Random Forest Regressor

### Boosting
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

### Stacking
A Stacking Regressor combining:
- Random Forest
- XGBoost
- LightGBM

Ridge Regression was used as the meta-learner.

### Hybrid Model
A hybrid approach combining:
- Random Forest
- Artificial Neural Network (ANN)

The predictions from the two models were combined using an equal-weight average.

## 4. Hyperparameter Tuning

RandomizedSearchCV was used to tune the Random Forest model.

The search used:
- 10 parameter combinations
- 3-fold cross-validation
- Negative RMSE as the scoring metric

The tuned model did not improve upon the baseline Random Forest on the test set. The baseline model was therefore retained as the better-performing Random Forest configuration.

## 5. Model Evaluation

The models were evaluated using three regression metrics:

- **MAE:** Measures the average absolute prediction error. Lower is better.
- **RMSE:** Penalizes larger prediction errors more heavily. Lower is better.
- **R²:** Measures the proportion of target variance explained by the model. Higher is better.

### Test Set Results

| Model | MAE (LKR) | RMSE (LKR) | R² |
|---|---:|---:|---:|
| Random Forest | 91,538.99 | 144,565.26 | 0.5962 |
| XGBoost | 94,623.25 | 141,070.76 | 0.6155 |
| LightGBM | 99,753.41 | 156,498.17 | 0.5268 |
| CatBoost | 102,292.12 | 147,768.62 | 0.5781 |
| Stacking | 92,993.64 | 141,023.89 | 0.6158 |
| Hybrid (RF + ANN) | 98,887.79 | 150,052.62 | 0.5650 |
| Tuned Random Forest | 92,579.22 | 144,742.70 | 0.5952 |

### Key Findings

- Random Forest achieved the lowest MAE, giving it the best average absolute prediction error.
- Stacking achieved the highest R² and lowest RMSE, although its improvement over XGBoost was small.
- XGBoost performed competitively, particularly in RMSE and R².
- LightGBM and CatBoost did not outperform the other approaches on this dataset.
- The hybrid RF + ANN model did not improve upon the standalone Random Forest.
- Hyperparameter tuning did not improve the baseline Random Forest's test performance.

The results demonstrate that combining models does not automatically guarantee better predictions. Performance depends on the dataset, model characteristics, and how well the learners complement one another.

### Feature Importance

Random Forest feature importance indicated that engine capacity, manufacturing year, and mileage were among the most influential predictive features.

Feature importance reflects predictive contribution and should not be interpreted as proof of causation.

## 6. Interactive Dashboard

The project includes a Streamlit dashboard with four sections:

1. **Overview:** Dataset statistics and ensemble learning approaches.
2. **Model Comparison:** Evaluation results using MAE, RMSE, and R².
3. **Feature Importance:** Visualization of the most influential Random Forest features.
4. **Live Prediction:** User input for bike specifications and price estimates from four trained models.

The live prediction interface currently serves Random Forest, XGBoost, LightGBM, and CatBoost. Stacking and Hybrid results are included in the evaluation comparison, but are not currently served through the live prediction interface.

## 7. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- TensorFlow / Keras
- Matplotlib and Seaborn
- Altair
- Streamlit
- Joblib

## 8. Running the Project Locally

### Requirements

- Python 3.12
- pip

### Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd used-bike-ensemble-dashboard
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the local dashboard at:

```text
http://localhost:8501
```

The `saved_models` directory must be present alongside `app.py`.

## 9. Repository Structure

```text
used-bike-ensemble-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
└── saved_models/
    ├── random_forest.pkl
    ├── xgboost_model.json
    ├── lightgbm.pkl
    ├── catboost.pkl
    ├── preprocessor.pkl
    ├── ann_preprocessor.pkl
    ├── ann_model.keras
    ├── stacking.pkl
    ├── feature_importance.csv
    └── model_results.csv
```

The original experimentation notebook should also be included in the repository.

## 10. AI Usage Acknowledgement

AI assistance was used for:
- Guidance on Python implementation and debugging.
- Assistance with Streamlit dashboard development and interface improvements.
- Help with code explanations, documentation, and README preparation.

## 11. Links

- **GitHub Repository:** https://github.com/Gethmi-Rathnayaka/used-bike-ensemble-dashboard
- **Live Dashboard:** https://used-bike-ensemble-dashboard-gasxqgns3xlhvhieyjitny.streamlit.app/

---

**Project:** Used Bike Price Prediction Using Ensemble Learning  
**Dataset:** [Sri Lankan used-bike listings from ikman.lk ](https://www.kaggle.com/datasets/ravinathwanni/used-bike-prices-in-sri-lanka) 