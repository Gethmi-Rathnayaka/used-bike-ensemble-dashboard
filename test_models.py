
import joblib
import os
import sklearn
import xgboost as xgb

print("Scikit-learn version:", sklearn.__version__)
print("XGBoost version:", xgb.__version__)

# Test models saved using joblib
model_files = [
    "random_forest.pkl",
    "lightgbm.pkl",
    "catboost.pkl",
    "preprocessor.pkl",
    "ann_preprocessor.pkl"
]

for filename in model_files:
    path = os.path.join("saved_models", filename)

    try:
        print(f"\nLoading {filename}...")
        obj = joblib.load(path)
        print(f"SUCCESS: {filename}")

    except Exception as e:
        print(f"FAILED: {filename}")
        print(type(e).__name__, str(e))


# Test XGBoost using its native JSON format
try:
    print("\nLoading xgboost_model.json...")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(
        os.path.join("saved_models", "xgboost_model.json")
    )

    print("SUCCESS: xgboost_model.json")

except Exception as e:
    print("FAILED: xgboost_model.json")
    print(type(e).__name__, str(e))


# Stacking still needs separate attention
print("\nStacking: Not tested yet (contains the old XGBoost pickle).")
