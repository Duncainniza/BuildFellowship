import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import kagglehub
import os

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
)

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(page_title="Farm Yield Analysis Dashboard", layout="wide", page_icon="🌾")

# ==========================================================
# Data loading + feature engineering (cached)
# ==========================================================
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("bhadramohit/agriculture-and-farming-dataset")
    csv_filename = "agriculture_dataset.csv"
    df = pd.read_csv(os.path.join(path, csv_filename))
    df.columns = [c.strip() for c in df.columns]

    # Per-acre efficiency features
    df["Fertilizer_per_acre"] = df["Fertilizer_Used(tons)"] / df["Farm_Area(acres)"]
    df["Pesticide_per_acre"] = df["Pesticide_Used(kg)"] / df["Farm_Area(acres)"]
    df["Water_per_acre"] = df["Water_Usage(cubic meters)"] / df["Farm_Area(acres)"]
    df["Yield_per_acre"] = df["Yield(tons)"] / df["Farm_Area(acres)"]

    median_yield_pa = df["Yield_per_acre"].median()
    df["High_Yield"] = (df["Yield_per_acre"] > median_yield_pa).astype(int)

    return df, median_yield_pa


@st.cache_resource
def build_preprocessor(df):
    target_reg = "Yield(tons)"
    target_clf = "High_Yield"
    drop_cols = ["Farm_ID", "Yield(tons)", "Yield_per_acre", "High_Yield"]
    features = [c for c in df.columns if c not in drop_cols]

    cat_features = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        sparse_threshold=0,
    )

    X_processed = preprocessor.fit_transform(df[features])
    feat_names = num_features + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
    )
    X = pd.DataFrame(X_processed, columns=feat_names)

    y_reg = df[target_reg]
    y_clf = df[target_clf]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)

    return {
        "features": features, "cat_features": cat_features, "num_features": num_features,
        "preprocessor": preprocessor, "feat_names": feat_names,
        "X": X, "y_reg": y_reg, "y_clf": y_clf,
        "X_train": X_train, "X_test": X_test,
        "y_reg_train": y_reg_train, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_test": y_clf_test,
    }


@st.cache_resource
def train_regression_models(_prep):
    X, X_train, X_test = _prep["X"], _prep["X_train"], _prep["X_test"]
    y_reg, y_reg_train, y_reg_test = _prep["y_reg"], _prep["y_reg_train"], _prep["y_reg_test"]

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        results[name] = {
            "Test R2": r2_score(y_reg_test, pred),
            "Test MAE": mean_absolute_error(y_reg_test, pred),
            "CV R2 (5-fold)": cross_val_score(model, X, y_reg, cv=5, scoring="r2").mean(),
        }
        fitted[name] = model
    return pd.DataFrame(results).T.round(3), fitted


@st.cache_resource
def train_classification_models(_prep):
    X, X_train, X_test = _prep["X"], _prep["X_train"], _prep["X_test"]
    y_clf, y_clf_train, y_clf_test = _prep["y_clf"], _prep["y_clf_train"], _prep["y_clf_test"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_clf_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results[name] = {
            "Test Accuracy": accuracy_score(y_clf_test, pred),
            "Test F1": f1_score(y_clf_test, pred),
            "Test ROC-AUC": roc_auc_score(y_clf_test, proba) if proba is not None else np.nan,
            "CV Accuracy (5-fold)": cross_val_score(model, X, y_clf, cv=5, scoring="accuracy").mean(),
        }
        fitted[name] = model
    return pd.DataFrame(results).T.round(3), fitted


@st.cache_resource
def tune_random_forest(_prep):
    X_train, y_clf_train = _prep["X_train"], _prep["y_clf_train"]
    X_test, y_clf_test = _prep["X_test"], _prep["y_clf_test"]
    X, y_clf = _prep["X"], _prep["y_clf"]

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 3, 5],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", None],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=1,
    )
    grid_search.fit(X_train, y_clf_train)
    best_rf = grid_search.best_estimator_
    pred = best_rf.predict(X_test)
    tuned_metrics = {
        "Best CV Accuracy": grid_search.best_score_,
        "Tuned Test Accuracy": accuracy_score(y_clf_test, pred),
        "Tuned Test F1": f1_score(y_clf_test, pred),
    }
    return grid_search.best_params_, tuned_metrics, best_rf


@st.cache_resource
def train_extra_models(_prep):
    X, X_train, X_test = _prep["X"], _prep["X_train"], _prep["X_test"]
    y_reg, y_reg_train, y_reg_test = _prep["y_reg"], _prep["y_reg_train"], _prep["y_reg_test"]
    y_clf, y_clf_train, y_clf_test = _prep["y_clf"], _prep["y_clf_train"], _prep["y_clf_test"]
    feat_names = _prep["feat_names"]

    lasso = Lasso(alpha=0.5, random_state=42)
    lasso.fit(X_train, y_reg_train)
    pred = lasso.predict(X_test)
    lasso_metrics = {
        "Test R2": r2_score(y_reg_test, pred),
        "Test MAE": mean_absolute_error(y_reg_test, pred),
        "CV R2 (5-fold)": cross_val_score(lasso, X, y_reg, cv=5, scoring="r2").mean(),
    }
    nonzero = pd.Series(lasso.coef_, index=feat_names)
    nonzero = nonzero[nonzero != 0].sort_values(key=abs, ascending=False)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_clf_train)
    pred = knn.predict(X_test)
    knn_metrics = {
        "Test Accuracy": accuracy_score(y_clf_test, pred),
        "Test F1": f1_score(y_clf_test, pred),
        "CV Accuracy (5-fold)": cross_val_score(knn, X, y_clf, cv=5, scoring="accuracy").mean(),
    }
    k_scores = {k: cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y_clf, cv=5, scoring="accuracy").mean()
                for k in range(1, 11)}
    k_series = pd.Series(k_scores, name="CV Accuracy").rename_axis("k")

    return lasso_metrics, nonzero, knn_metrics, k_series


# ==========================================================
# Load everything once
# ==========================================================
with st.spinner("Loading data and training models..."):
    df, median_yield_pa = load_data()
    prep = build_preprocessor(df)
    reg_table, reg_models = train_regression_models(prep)
    clf_table, clf_models = train_classification_models(prep)
    best_params, tuned_metrics, best_rf = tune_random_forest(prep)
    lasso_metrics, lasso_nonzero, knn_metrics, k_series = train_extra_models(prep)

# ==========================================================
# Sidebar navigation
# ==========================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "EDA", "Preprocessing", "Modelling", "Model Comparison", "Predict", "Insights"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"You're on the **{page}** page")

# ==========================================================
# HOME
# ==========================================================
if page == "Home":
    st.title("🌾 Farm Yield Analysis Dashboard")
    st.markdown("Welcome to the Farm Yield Analysis Dashboard. This project analyzes the "
                "**Agriculture & Farming dataset** (50 farms) to understand which factors "
                "actually drive crop productivity.")


    st.subheader("🎯 Business Problem & Objective")
    st.markdown(
        "Farm productivity (yield) depends on a mix of controllable inputs and practice "
        "choices fertilizer, pesticide, water, irrigation method, crop type, soil type, "
        "and season but the relationship between these factors and productivity isn't "
        "well understood at the farm-operator level. This dashboard builds models to:\n\n"
        "1. **Predict** estimate a farm's yield (or productivity tier) from its characteristics.\n"
        "2. **Identify drivers** determine which factors matter most, so recommendations can be "
        "made about where to focus resources.\n\n"
        "**A note on scope:** this dataset has only 50 rows, so every model here should be "
        "treated as exploratory and directional rather than production-grade. 5-fold "
        "cross-validation is used throughout, and results are reported honestly including "
        "where a model doesn't work rather than overstating what the data supports."
    )

    st.subheader("🗂️ Pages in this Dashboard")
    st.markdown(
        "- **EDA**: visual exploration of features and how they relate to yield\n"
        "- **Preprocessing**: data cleaning, feature engineering, and the transformation pipeline\n"
        "- **Modelling**: training and evaluating regression & classification models\n"
        "- **Model Comparison**: side-by-side comparison, hyperparameter tuning, and two extra models\n"
        "- **Predict**: try it yourself enter farm characteristics and get a live prediction\n"
        "- **Insights**: key takeaways and recommendations"
    )

    st.subheader("📁 Dataset Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows (farms)", df.shape[0])
        st.metric("Columns", df.shape[1])
    with c2:
        st.write("**Columns:**")
        st.write(list(df.columns))

    st.write("**Preview:**")
    st.dataframe(df.head())

# ==========================================================
# EDA
# ==========================================================
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("The goal of this section is to visually check: do any of these factors look "
                "like they move yield up or down? This gives a first read on the business "
                "question which factors influence productivity before building any models.")

    st.subheader("Distributions of Key Numeric Features")
    num_cols = ["Farm_Area(acres)", "Fertilizer_Used(tons)", "Pesticide_Used(kg)", "Yield(tons)"]
    fig = px.histogram(df, x=num_cols[0], marginal="box", template="plotly_dark")
    sel_col = st.selectbox("Choose a numeric feature to inspect", num_cols)
    fig = px.histogram(df, x=sel_col, marginal="box", nbins=15,
                        title=f"Distribution of {sel_col}", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What this chart shows:** a histogram groups farms into bins and shows how many "
        "farms fall into each bin for that measurement.\n\n"
        "**What it means:** farm area, fertilizer use, pesticide use, and yield are all "
        "fairly spread out across their range rather than clustering tightly around one "
        "typical value there's no single 'typical' farm in this dataset. None of the "
        "distributions show an obvious skew or outlier problem that would need fixing "
        "before modelling."
    )

    st.subheader("Correlation Matrix (Numeric Features)")
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     aspect="auto", template="plotly_dark", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What this chart shows:** a correlation matrix measures how strongly two numeric "
        "variables move together, from -1 (perfectly opposite) to +1 (perfectly together); "
        "0 means no relationship.\n\n"
        "**What it means for the business question:** look at the `Yield(tons)` row/column "
        "specifically every number there is close to 0. In plain terms, **none of the raw "
        "numeric inputs (farm area, fertilizer, pesticide, water) has a strong straight-line "
        "relationship with yield on its own.** This is an early warning that predicting exact "
        "yield from these features alone may be difficult."
    )

    st.subheader("Yield by Category")
    cat_option = st.selectbox("Choose a category", ["Irrigation_Type", "Soil_Type", "Season", "Crop_Type"])
    fig = px.box(df, x=cat_option, y="Yield(tons)", template="plotly_dark",
                 title=f"Yield by {cat_option}")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What this chart shows:** each box summarizes the spread of yield values for one "
        "category. The line in the middle is the median (typical) yield for that group, the "
        "box covers the middle 50% of farms, and the whiskers show the rest of the range.\n\n"
        "**What it means:** the boxes overlap heavily across categories no irrigation "
        "method, soil type, season, or crop stands out as a clear yield winner or loser in "
        "this data. Combined with the flat correlation matrix, this points to a **low "
        "signal-to-noise ratio** for the raw yield number, which carries forward into the "
        "modelling results below."
    )

# ==========================================================
# PREPROCESSING
# ==========================================================
elif page == "Preprocessing":
    st.title("🧹 Preprocessing & Feature Engineering")

    st.subheader("Data Quality")
    st.write(f"Missing values: **{df.drop(columns=['High_Yield']).isnull().sum().sum()}**  |  "
             f"Duplicate rows: **{df.duplicated().sum()}**")
    st.markdown("No missing values and no duplicates — the dataset is already clean, so there's "
                "minimal cleaning work here (this is a synthetically generated Kaggle dataset, "
                "which explains why it's so tidy).")

    st.subheader("Feature Engineering")
    st.markdown(
        "Based on the EDA, **per-acre efficiency features** were engineered (fertilizer, "
        "pesticide, and water use per acre), since a 400-acre farm and a 20-acre farm using "
        "the same *total* tons of fertilizer aren't really comparable — what matters for "
        "efficiency is intensity of use, not the raw total. A `High_Yield` binary target was "
        "also created for the classification task: is this farm more or less productive than "
        "a typical farm, per acre?"
    )
    st.code(
        "df['Fertilizer_per_acre'] = df['Fertilizer_Used(tons)'] / df['Farm_Area(acres)']\n"
        "df['Pesticide_per_acre']  = df['Pesticide_Used(kg)'] / df['Farm_Area(acres)']\n"
        "df['Water_per_acre']      = df['Water_Usage(cubic meters)'] / df['Farm_Area(acres)']\n"
        "df['Yield_per_acre']      = df['Yield(tons)'] / df['Farm_Area(acres)']\n\n"
        "median_yield_pa = df['Yield_per_acre'].median()\n"
        "df['High_Yield'] = (df['Yield_per_acre'] > median_yield_pa).astype(int)",
        language="python",
    )
    st.write(f"Median yield per acre used as the split threshold: **{round(median_yield_pa, 3)} tons/acre**")
    st.write("High_Yield class balance:")
    st.dataframe(df["High_Yield"].value_counts().rename("count"))

    st.subheader("Feature Selection")
    st.markdown(
        "All remaining features are kept in for the baseline models — with only 50 rows and "
        "~8 raw features, there isn't a strong case for dropping anything up front. Model "
        "coefficients / feature importances are inspected *after* fitting to see which "
        "features actually pull weight (see the Model Comparison page)."
    )
    st.write("**Categorical features:**", prep["cat_features"])
    st.write("**Numeric features:**", prep["num_features"])

    st.subheader("Transformation Pipeline")
    st.markdown(
        "Numeric features are scaled with `RobustScaler` (robust to outliers, uses median "
        "and IQR rather than mean/std), and categorical features are one-hot encoded with "
        "`OneHotEncoder`. The two are combined with a `ColumnTransformer`."
    )
    st.code(
        "preprocessor = ColumnTransformer(transformers=[\n"
        "    ('num', RobustScaler(), num_features),\n"
        "    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),\n"
        "], sparse_threshold=0)",
        language="python",
    )
    st.write(f"Resulting feature matrix shape: **{prep['X'].shape}**")
    st.write(f"Train / test split: **{prep['X_train'].shape[0]} / {prep['X_test'].shape[0]}** "
             "farms (70/30 split, `random_state=42`)")

# ==========================================================
# MODELLING
# ==========================================================
elif page == "Modelling":
    st.title("🤖 Modelling")

    st.subheader("Regression: Predicting `Yield(tons)`")
    st.markdown("This directly targets business sub-goal #1: predicting yield. `R²` tells us "
                "what share of the variation in yield the model explains (1.0 = perfect, "
                "0 = no better than guessing the average, negative = worse than guessing the average).")
    st.dataframe(reg_table)
    st.info(
        "**What this table means:** every model's R² is **negative**. In plain language, a "
        "negative R² means the model does *worse* than the simplest possible baseline just "
        "guessing the average yield for every farm. This is a real and important finding, not "
        "a coding error: **with this feature set and only 50 farms, exact yield in tons cannot "
        "be reliably predicted.** This likely reflects either (a) yield being close to random "
        "noise in this synthetic dataset, (b) too little data for the number of features used, "
        "or (c) real-world yield depending on factors not captured here at all (weather, pest "
        "pressure, timing, soil nutrients)."
    )

    st.subheader("Classification: Predicting `High_Yield`")
    st.markdown("Since predicting exact tons didn't work, this reframes the business question in "
                "a more learnable way: instead of 'exactly how many tons will this farm yield,' "
                "we ask **'is this farm more or less productive than a typical farm, per acre?'** "
                "This removes the effect of farm size and focuses purely on efficiency.")
    st.dataframe(clf_table)
    st.info(
        "**What this table means:** unlike the regression results, these numbers are "
        "meaningfully better than random guessing (50% accuracy would be the baseline for a "
        "50/50 split). **Random Forest is the strongest of the three**, with the best "
        "cross-validated accuracy confirming that reframing 'predict yield' as 'predict "
        "relative productivity' is the more workable version of the business question for "
        "this dataset."
    )

    st.subheader("Confusion Matrices (Test Set)")
    model_choice = st.selectbox("Choose a classifier", list(clf_models.keys()))
    model = clf_models[model_choice]
    cm = confusion_matrix(prep["y_clf_test"], model.predict(prep["X_test"]))
    fig = px.imshow(cm, text_auto=True, x=["Pred: Low", "Pred: High"], y=["True: Low", "True: High"],
                     color_continuous_scale="Blues", template="plotly_dark",
                     title=f"Confusion Matrix — {model_choice}")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What this chart shows:** the top-left and bottom-right squares are correct "
        "predictions; the top-right and bottom-left squares are mistakes.\n\n"
        "**What it means:** more weight on the diagonal means fewer mistakes. Random Forest "
        "typically shows the most weight on the diagonal on this dataset with only 35 "
        "training farms, a single Decision Tree tends to overfit to quirks of the training "
        "data, which shows up as more off-diagonal errors. Random Forest's trick of averaging "
        "many trees together reduces that problem."
    )

# ==========================================================
# MODEL COMPARISON
# ==========================================================
elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")

    st.subheader("Regression vs. Classification — CV Score Comparison")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(reg_table.reset_index(), x="index", y="CV R2 (5-fold)",
                      title="Regression Models — CV R²", template="plotly_dark",
                      labels={"index": "Model"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(clf_table.reset_index(), x="index", y="CV Accuracy (5-fold)",
                      title="Classification Models — CV Accuracy", template="plotly_dark",
                      labels={"index": "Model"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which Factors Actually Drive Productivity?")
    st.markdown("Three independent views into the same models, used to answer business "
                "sub-goal #2: identifying which factors influence productivity.")

    log_reg = clf_models["Logistic Regression"]
    coefs = pd.Series(log_reg.coef_[0], index=prep["feat_names"]).sort_values(key=abs, ascending=False)
    fig = px.bar(coefs.head(10).sort_values(), orientation="h", template="plotly_dark",
                 title="Logistic Regression — Top 10 Coefficients (log-odds) for High_Yield",
                 labels={"value": "Coefficient (positive = pushes toward High Yield)", "index": "Feature"})
    st.plotly_chart(fig, use_container_width=True)

    rf_clf = clf_models["Random Forest"]
    importances = pd.Series(rf_clf.feature_importances_, index=prep["feat_names"]).sort_values(ascending=False)
    fig = px.bar(importances.head(10).sort_values(), orientation="h", template="plotly_dark",
                 title="Random Forest — Top 10 Feature Importances (High_Yield)",
                 labels={"value": "Importance", "index": "Feature"})
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**What these charts mean:** the longest bars in both charts belong mostly to the "
        "per-acre efficiency features (fertilizer, pesticide, and water per acre), not to "
        "which specific crop, soil type, or irrigation method was used. Having two different "
        "types of models agree independently makes this a more trustworthy conclusion: "
        "**input efficiency (intensity per acre), not crop choice or irrigation method, is "
        "the strongest driver of relative productivity in this dataset.**"
    )

    st.subheader("Decision Tree Structure (Top 3 Levels)")
    dt_clf = clf_models["Decision Tree"]
    fig_tree, ax = plt.subplots(figsize=(16, 8))
    plot_tree(dt_clf, feature_names=prep["feat_names"], class_names=["Low", "High"],
              filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax)
    st.pyplot(fig_tree)
    st.info(
        "**What this shows:** the actual decision-making logic of the single Decision Tree, "
        "drawn as a flowchart. Each split asks a yes/no question about a feature and routes a "
        "farm left or right. The features chosen for the earliest (top) splits are generally "
        "the ones the tree found most useful for separating Low vs. High yield farms — a "
        "useful sanity check against the two importance charts above."
    )

    st.subheader("Hyperparameter Tuning (GridSearchCV on Random Forest)")
    st.markdown("Random Forest was the strongest classifier above, so it's the one worth "
                "tuning further with `GridSearchCV`, which systematically tries different "
                "model settings and keeps the combination that performs best under "
                "cross-validation.")
    st.write("**Best hyperparameters found:**")
    st.json(best_params)
    st.dataframe(pd.Series(tuned_metrics, name="Score").to_frame().round(3))
    st.info(
        "**What it means:** compare 'Best Cross-Validated Accuracy' above to the untuned "
        "Random Forest's CV accuracy on the Modelling page. With a dataset this small, don't "
        "expect a dramatic jump even a small drop is possible, since the search is "
        "optimizing on the same handful of folds it's being scored on. This is a reminder for "
        "why cross-validated scores, not a single test-set score, should drive which model "
        "gets used in practice."
    )

    st.subheader("Two Additional Models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Lasso Regression** built-in feature selection; can shrink weak "
                     "factors' influence all the way to zero.")
        st.dataframe(pd.Series(lasso_metrics, name="Score").to_frame().round(3))
        st.write(f"Features kept non-zero ({len(lasso_nonzero)} of {len(prep['feat_names'])}):")
        st.dataframe(lasso_nonzero.round(3))
    with c2:
        st.markdown("**K-Nearest Neighbors (k=5)** — classifies a farm by majority vote of "
                     "its *k* most similar farms.")
        st.dataframe(pd.Series(knn_metrics, name="Score").to_frame().round(3))
        fig = px.line(k_series.reset_index(), x="k", y="CV Accuracy", markers=True,
                      template="plotly_dark", title="KNN — CV Accuracy vs. k")
        st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What it means:** Lasso zeroes out most of the crop/soil/season/irrigation category "
        "columns entirely, agreeing with the flat correlation matrix it can't find enough "
        "signal in most of those features to justify keeping them. Its R² is still negative, "
        "a third independent confirmation that exact yield prediction isn't supported by this "
        "data. KNN's accuracy is noticeably below Random Forest's and shifts around as *k* "
        "changes a sign it isn't finding a stable pattern, expected here since with ~30 "
        "features but only 50 farms, the idea of 'similar farms' becomes unreliable (the "
        "*curse of dimensionality*). Tree-based models are the better-suited family for this "
        "dataset's size and shape."
    )

# ==========================================================
# PREDICT
# ==========================================================
elif page == "Predict":
    st.title("🔮 Predict")
    st.markdown("Enter a hypothetical farm's characteristics below to get a live prediction "
                "from the tuned Random Forest classifier (for productivity tier) and the "
                "Random Forest regressor (for exact yield, shown with its important caveat).")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            crop_type = st.selectbox("Crop Type", sorted(df["Crop_Type"].unique()))
            irrigation_type = st.selectbox("Irrigation Type", sorted(df["Irrigation_Type"].unique()))
            soil_type = st.selectbox("Soil Type", sorted(df["Soil_Type"].unique()))
            season = st.selectbox("Season", sorted(df["Season"].unique()))
        with c2:
            farm_area = st.slider("Farm Area (acres)", float(df["Farm_Area(acres)"].min()),
                                   float(df["Farm_Area(acres)"].max()), float(df["Farm_Area(acres)"].median()))
            fertilizer = st.slider("Fertilizer Used (tons)", float(df["Fertilizer_Used(tons)"].min()),
                                    float(df["Fertilizer_Used(tons)"].max()), float(df["Fertilizer_Used(tons)"].median()))
            pesticide = st.slider("Pesticide Used (kg)", float(df["Pesticide_Used(kg)"].min()),
                                   float(df["Pesticide_Used(kg)"].max()), float(df["Pesticide_Used(kg)"].median()))
            water = st.slider("Water Usage (cubic meters)", float(df["Water_Usage(cubic meters)"].min()),
                               float(df["Water_Usage(cubic meters)"].max()), float(df["Water_Usage(cubic meters)"].median()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([{
            "Crop_Type": crop_type,
            "Farm_Area(acres)": farm_area,
            "Irrigation_Type": irrigation_type,
            "Fertilizer_Used(tons)": fertilizer,
            "Pesticide_Used(kg)": pesticide,
            "Soil_Type": soil_type,
            "Season": season,
            "Water_Usage(cubic meters)": water,
        }])
        row["Fertilizer_per_acre"] = row["Fertilizer_Used(tons)"] / row["Farm_Area(acres)"]
        row["Pesticide_per_acre"] = row["Pesticide_Used(kg)"] / row["Farm_Area(acres)"]
        row["Water_per_acre"] = row["Water_Usage(cubic meters)"] / row["Farm_Area(acres)"]

        row_transformed = prep["preprocessor"].transform(row[prep["features"]])
        row_X = pd.DataFrame(row_transformed, columns=prep["feat_names"])

        clf_pred = best_rf.predict(row_X)[0]
        clf_proba = best_rf.predict_proba(row_X)[0][1]
        reg_pred = reg_models["Random Forest"].predict(row_X)[0]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Productivity Tier",
                       "High Yield" if clf_pred == 1 else "Low Yield",
                       delta=f"{clf_proba:.0%} confidence")
        with c2:
            st.metric("Predicted Exact Yield (tons)", f"{reg_pred:.1f}")

        st.warning(
            "**Caveat:** the exact-tons regression models all scored a negative R² during "
            "evaluation, meaning they perform worse than simply guessing the average yield. "
            "Treat the exact-tons number above as illustrative only the productivity-tier "
            "(High/Low Yield) prediction is the more trustworthy of the two, since the "
            "classification models scored meaningfully above random guessing."
        )

# ==========================================================
# INSIGHTS
# ==========================================================
elif page == "Insights":
    st.title("💡 Insights & Recommendations")
    st.markdown(
        "**Data quality:** the cleaning process confirmed there were no missing values or "
        "duplicate records this is a synthetically generated Kaggle dataset, which explains "
        "why it's so tidy.\n\n"
        "**Predicting exact yield doesn't work here.** Every regression model tested "
        "(Linear, Decision Tree, Random Forest, Lasso) produced a negative R² on this "
        "50-farm dataset worse than simply guessing the average yield for every farm. This "
        "likely reflects either yield being close to random noise in this synthetic dataset, "
        "too little data for the number of features used, or real-world yield depending on "
        "factors not captured here at all (weather, pest pressure, timing, soil nutrients).\n\n"
        "**Reframing as a classification problem works much better.** Predicting whether a "
        "farm is above or below the median yield-per-acre (`High_Yield`) is meaningfully "
        "better than random guessing, with **Random Forest the strongest performer**.\n\n"
        "**Input efficiency, not crop or method choice, drives relative productivity.** Both "
        "Logistic Regression coefficients and Random Forest feature importances — two "
        "independent model types agree that the engineered per-acre features (fertilizer, "
        "pesticide, water intensity) matter more than which crop is grown, which irrigation "
        "method is used, which soil type, or which season. Lasso Regression independently "
        "confirms this by zeroing out most of the category columns entirely.\n\n"
        "**Model family matters at this data size.** Tree-based ensembles (Random Forest) "
        "consistently outperform both a single Decision Tree (which overfits with only 35 "
        "training farms) and KNN (which suffers from the curse of dimensionality with ~30 "
        "features but only 50 farms).\n\n"
        "**Recommendation:** rather than chasing exact yield predictions from this dataset, "
        "resources are better spent (a) collecting more farms and/or more informative "
        "features (weather, soil nutrients, pest pressure, timing), and (b) in the meantime, "
        "using the High/Low Yield classifier and its efficiency-per-acre drivers to flag "
        "farms that may benefit from adjusting input intensity regardless of what crop "
        "they grow or how they're irrigated."
    )
