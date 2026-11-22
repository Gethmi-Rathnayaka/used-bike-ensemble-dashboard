import os
import altair as alt
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Used Bike Price Prediction",
    page_icon="🛵",
    layout="wide",
)

# ---------------- THEME & PALETTE ----------------

PALETTE = {
    "coral": "#E86F58",       # Primary accent / best model highlight
    "coral_soft": "#FDECE8",  # Soft banner / card tint
    "sky": "#5C9EAD",         # Secondary chart bars
    "mint": "#4EA898",        # Prediction chart & positive highlights
    "plum": "#6D597A",        # Feature importance bars
    "sand_border": "#EADCD0",
    "ink": "#2B2D42",
}

st.markdown(
    f"""
    <style>
    .hero-banner {{
        background: linear-gradient(135deg, {PALETTE["coral_soft"]} 0%, #EEF6F8 100%);
        border: 1px solid {PALETTE["sand_border"]};
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .hero-banner h1 {{
        margin: 0;
        font-size: 2rem;
        color: {PALETTE["ink"]};
    }}
    .hero-banner p {{
        margin: 0.35rem 0 0 0;
        color: #555B6E;
        font-size: 0.98rem;
    }}
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid {PALETTE["sand_border"]};
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }}
    div[data-testid="stMetricValue"] {{
        color: {PALETTE["coral"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = "saved_models"

PRESETS = {
    "Honda Hornet (2020)": {
        "bike_type": "Motorbikes",
        "brand": "Honda",
        "trim": "Unknown",
        "model": "Hornet",
        "year": 2020,
        "mileage": 15000,
        "capacity": 150,
    },
    "Daily Commuter Scooter": {
        "bike_type": "Scooters",
        "brand": "TVS",
        "trim": "NTORQ",
        "model": "NTORQ 125",
        "year": 2022,
        "mileage": 12000,
        "capacity": 125,
    },
    "Older High-Mileage Bike": {
        "bike_type": "Motorbikes",
        "brand": "Bajaj",
        "trim": "Standard",
        "model": "Pulsar 150",
        "year": 2014,
        "mileage": 68000,
        "capacity": 150,
    },
}

DEFAULT_STATE = {
    "bike_type": "Motorbikes",
    "brand": "Honda",
    "trim": "Unknown",
    "model": "Hornet",
    "year": 2020,
    "mileage": 15000,
    "capacity": 150,
}

for key, val in DEFAULT_STATE.items():
    st.session_state.setdefault(key, val)


# ---------------- LOAD MODELS AND DATA ----------------


@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl")),
        "CatBoost": joblib.load(os.path.join(MODEL_DIR, "catboost.pkl")),
    }

    models["XGBoost"].load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))

    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

    return models, preprocessor


@st.cache_data
def load_data():
    results = pd.read_csv(os.path.join(MODEL_DIR, "model_results.csv"))
    importance = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))
    return results, importance


def apply_preset(preset_values: dict):
    st.session_state["bike_type"] = preset_values["bike_type"]
    st.session_state["brand"] = preset_values["brand"]
    st.session_state["trim"] = preset_values["trim"]
    st.session_state["model"] = preset_values["model"]
    st.session_state["year"] = preset_values["year"]
    st.session_state["mileage"] = preset_values["mileage"]
    st.session_state["capacity"] = preset_values["capacity"]


def make_metric_chart(df: pd.DataFrame, metric_col: str, title: str, x_title: str, ascending_best: bool = True):
    best_val = df[metric_col].min() if ascending_best else df[metric_col].max()
    chart_df = df.copy()
    chart_df["Status"] = chart_df[metric_col].apply(
        lambda v: "Best" if v == best_val else "Other"
    )

    order_sort = "-x" if ascending_best else "x"

    return (
        alt.Chart(chart_df, title=title)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X(f"{metric_col}:Q", title=x_title),
            y=alt.Y("Model:N", sort=order_sort, title=None),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Best", "Other"],
                    range=[PALETTE["coral"], PALETTE["sky"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(f"{metric_col}:Q", format=",.2f" if metric_col != "R2" else ".4f"),
            ],
        )
        .properties(height=240)
        .configure_view(strokeWidth=0)
    )


try:
    models, preprocessor = load_models()
    results_df, feature_importance_df = load_data()
except Exception as e:
    st.error(f"Error loading dashboard files: {e}")
    st.stop()


# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>Used Bike Price Prediction Dashboard</h1>
        <p>Ensemble machine learning models trained on ikman.lk listings</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, comparison_tab, importance_tab, prediction_tab = st.tabs([
    "Overview",
    "Model Comparison",
    "Feature Importance",
    "Live Prediction",
])


# ---------------- OVERVIEW ----------------

with overview_tab:
    st.header("Project Overview")
    st.write(
        "This dashboard presents an evaluation of ensemble learning techniques "
        "for predicting used bike prices (in LKR) from Sri Lankan marketplace listings."
    )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset Size", "5,016 listings")
        col2.metric("Models Evaluated", "7")
        col3.metric("Live Models Served", f"{len(models)}")
        col4.metric("Target Currency", "LKR")

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.subheader("Ensemble Approaches Evaluated")
            st.markdown("""
- **Bagging:** Random Forest
- **Boosting:** XGBoost, LightGBM, and CatBoost
- **Stacking:** Random Forest, XGBoost, and LightGBM with Ridge Regression as meta-learner
- **Hybrid:** Random Forest combined with an Artificial Neural Network

*The 4 standalone tree-based models are served directly in the live predictor for fast interactive inference.*
            """)

    with right_col:
        with st.container(border=True):
            st.subheader("Input Features & Metrics")
            st.markdown("""
**Features:** Bike Type, Brand, Trim/Edition, Model, Year, Mileage, and Capacity.

- **MAE:** Average absolute prediction error (lower is better).
- **RMSE:** Penalizes larger prediction errors more heavily (lower is better).
- **R²:** Proportion of price variance explained by the model (higher is better).
            """)


# ---------------- MODEL COMPARISON ----------------

with comparison_tab:
    st.header("Model Performance Comparison")
    st.write("Test-set evaluation across all trained models. Coral highlights the strongest score per metric.")

    with st.container(border=True):
        st.subheader("Evaluation Metrics Table")
        st.dataframe(
            results_df.style.format({
                "MAE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "R2": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.altair_chart(
                make_metric_chart(results_df, "MAE", "Mean Absolute Error (LKR)", "MAE (LKR)", ascending_best=True),
                use_container_width=True,
            )
            st.caption("Lower MAE indicates better average prediction accuracy.")

    with c2:
        with st.container(border=True):
            st.altair_chart(
                make_metric_chart(results_df, "RMSE", "Root Mean Squared Error (LKR)", "RMSE (LKR)", ascending_best=True),
                use_container_width=True,
            )
            st.caption("Lower RMSE indicates fewer large prediction errors.")

    with st.container(border=True):
        st.altair_chart(
            make_metric_chart(results_df, "R2", "R² Score (Test Fit)", "R²", ascending_best=False),
            use_container_width=True,
        )
        st.caption("Higher R² indicates a stronger fit to held-out test prices.")


# ---------------- FEATURE IMPORTANCE ----------------

with importance_tab:
    st.header("Random Forest Feature Importance")
    st.write("Top 15 features ranked by their predictive contribution in the Random Forest model.")

    top_features = (
        feature_importance_df
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fi_chart = (
        alt.Chart(top_features, title="Top 15 Most Influential Features")
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color=PALETTE["plum"])
        .encode(
            x=alt.X("Importance:Q", title="Importance Score"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Importance:Q", format=".4f"),
            ],
        )
        .properties(height=420)
        .configure_view(strokeWidth=0)
    )

    with st.container(border=True):
        st.altair_chart(fi_chart, use_container_width=True)

    with st.container(border=True):
        st.subheader("Key Takeaways")
        st.markdown("""
- **Engine capacity** leads overall influence on bike valuation.
- **Manufacturing year** and **mileage** capture age and wear-and-tear effects.
- **Brand and model** refine segment-specific pricing tiers.

*Feature importance reflects predictive contribution in the trained model, not causal impact.*
        """)


# ---------------- LIVE PREDICTION ----------------

with prediction_tab:
    st.header("Predict a Used Bike's Price")
    st.write("Pick a quick sample preset for demo testing or enter custom listing specifications.")

    st.markdown("**Quick Demo Presets:**")
    preset_cols = st.columns(len(PRESETS))
    for idx, (preset_label, preset_data) in enumerate(PRESETS.items()):
        with preset_cols[idx]:
            st.button(
                preset_label,
                key=f"preset_{idx}",
                use_container_width=True,
                on_click=apply_preset,
                args=(preset_data,),
            )

    with st.container(border=True):
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                bike_type = st.selectbox(
                    "Bike Type",
                    ["Motorbikes", "Scooters", "Quadricycles", "E-bikes"],
                    key="bike_type",
                )
                brand = st.text_input("Brand", key="brand")
                trim = st.text_input("Trim / Edition", key="trim")
                model_name = st.text_input("Model", key="model")

            with col2:
                year = st.number_input(
                    "Manufacturing Year",
                    min_value=1950,
                    max_value=2026,
                    key="year",
                )
                mileage = st.number_input(
                    "Mileage (km)",
                    min_value=0,
                    key="mileage",
                )
                capacity = st.number_input(
                    "Engine Capacity (cc)",
                    min_value=0,
                    key="capacity",
                )

            submitted = st.form_submit_button(
                "Predict Bike Price",
                use_container_width=True,
                type="primary",
            )

    if submitted:
        input_data = pd.DataFrame([{
            "Bike Type": bike_type,
            "Brand": brand,
            "Trim/Edition": trim,
            "Model": model_name,
            "Year": year,
            "Mileage": mileage,
            "Capacity": capacity,
        }])

        try:
            processed_data = preprocessor.transform(input_data)

            predictions = {
                name: float(model.predict(processed_data)[0])
                for name, model in models.items()
            }

            avg_price = sum(predictions.values()) / len(predictions)
            min_price = min(predictions.values())
            max_price = max(predictions.values())

            with st.container(border=True):
                st.subheader("Prediction Summary")

                s1, s2, s3 = st.columns(3)
                s1.metric("Average Estimate", f"LKR {avg_price:,.2f}")
                s2.metric("Lowest Estimate", f"LKR {min_price:,.2f}")
                s3.metric("Highest Estimate", f"LKR {max_price:,.2f}")

                st.divider()

                m_cols = st.columns(4)
                for idx, (name, price) in enumerate(predictions.items()):
                    m_cols[idx].metric(label=name, value=f"LKR {price:,.2f}")

                pred_df = pd.DataFrame([
                    {"Model": name, "Predicted Price (LKR)": price}
                    for name, price in predictions.items()
                ])

                pred_chart = (
                    alt.Chart(pred_df, title="Model-by-Model Price Estimates")
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color=PALETTE["mint"])
                    .encode(
                        x=alt.X("Predicted Price (LKR):Q", title="Estimated Price (LKR)"),
                        y=alt.Y("Model:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("Model:N"),
                            alt.Tooltip("Predicted Price (LKR):Q", format=",.2f"),
                        ],
                    )
                    .properties(height=200)
                    .configure_view(strokeWidth=0)
                )

                st.altair_chart(pred_chart, use_container_width=True)
                st.caption(
                    "These are model estimates based on historical listings, not guaranteed market prices."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")