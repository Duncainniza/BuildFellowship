import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import kagglehub

# ----------------------
# Page Config
# ----------------------
st.set_page_config(page_title="Traffic Violations Dashboard", layout="wide")

st.title("🚦 Traffic Violations Dashboard")
st.caption("Interactive EDA — converted from Week 3 notebook | Duncain Sichande")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode", ["Interactive EDA"])


# ----------------------
# Data loading + cleaning (cached)
# ----------------------
@st.cache_data
def get_data():
    path = kagglehub.dataset_download("nikhil1e9/traffic-violations")
    df = pd.read_csv(f"{path}/traffic_violations.csv")

    # Replace "." with "_" for easier attribute-style access
    df.columns = df.columns.str.replace(".", "_", regex=False)

    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Clean Year: keep only plausible vehicle model years, everything else becomes missing
    invalid_year_mask = (df["Year"] < 1950) | (df["Year"] > 2026)
    df.loc[invalid_year_mask, "Year"] = np.nan
    df["Year"] = df["Year"].astype("Int64")

    # 3. Standardize placeholder codes: "XX" -> missing for State-type columns
    for col in ["State", "DL_State"]:
        df.loc[df[col] == "XX", col] = np.nan

    # 4. Standardize inconsistent Color abbreviations
    color_map = {
        "BLUE DARK": "DARK BLUE",
        "BLUE LIGHT": "LIGHT BLUE",
        "GREEN DK": "DARK GREEN",
        "GREEN LGT": "LIGHT GREEN",
    }
    df["Color"] = df["Color"].replace(color_map)

    # 5. Strip stray whitespace from all text columns
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda col: col.str.strip())

    return df


@st.cache_data
def get_scatter_plot(data, x_axis, y_axis, color):
    fig = px.scatter(
        data, x=x_axis, y=y_axis, color=color,
        title=f"{x_axis} vs {y_axis}",
        template="plotly_dark",
    )
    return fig


@st.cache_data
def get_hist(data, hist_column, color_col):
    fig = px.histogram(
        data, x=hist_column, color=color_col, marginal="box",
        title=f"Distribution of {hist_column}",
        template="plotly_dark",
    )
    return fig


@st.cache_data
def get_box_plot(data, x_col, y_col):
    fig = px.box(
        data, x=x_col, y=y_col,
        title=f"{y_col} by {x_col}",
        template="plotly_dark",
    )
    return fig


@st.cache_data
def get_bar_plot(data, cat_column, top_n):
    counts = data[cat_column].value_counts().head(top_n)
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        labels={"x": "Number of Stops", "y": cat_column},
        title=f"Top {top_n} categories: {cat_column}",
        template="plotly_dark",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


@st.cache_data
def get_stacked_bar(data, cat_col, hue_col):
    cross = pd.crosstab(data[cat_col], data[hue_col], normalize="index")
    cross_long = cross.reset_index().melt(id_vars=cat_col, var_name=hue_col, value_name="proportion")
    fig = px.bar(
        cross_long, x=cat_col, y="proportion", color=hue_col,
        title=f"{hue_col} Proportions by {cat_col}",
        template="plotly_dark",
    )
    return fig


@st.cache_data
def get_pie_chart(data, cat_col):
    counts = data[cat_col].value_counts()
    fig = px.pie(
        values=counts.values, names=counts.index,
        title=f"{cat_col} Composition",
        template="plotly_dark",
    )
    return fig


@st.cache_data
def get_line_trend(data, year_min, year_max):
    subset = data[(data["Year"] >= year_min) & (data["Year"] <= year_max)]
    year_trend = subset["Year"].dropna().astype(int).value_counts().sort_index()
    fig = px.line(
        x=year_trend.index, y=year_trend.values, markers=True,
        labels={"x": "Vehicle Model Year", "y": "Number of Stops"},
        title=f"Number of Stops by Vehicle Model Year ({year_min}-{year_max})",
        template="plotly_dark",
    )
    return fig


@st.cache_data
def get_corr_heatmap(data, binary_cols):
    encoded = data[binary_cols].apply(lambda col: col.map({"Yes": 1, "No": 0}))
    corr = encoded.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", origin="lower",
        title="Correlation Between Safety/Accident Flags",
        template="plotly_dark",
    )
    return fig


# ----------------------
# App
# ----------------------
if app_mode == "Interactive EDA":

    with st.spinner("Loading dataset..."):
        df = get_data()

    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Dataset Dimensions:", df.shape)

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all").T)

    # Column groups
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    binary_cols = ["Belts", "Personal_Injury", "Property_Damage",
                   "Commercial_License", "Commercial_Vehicle", "Contributed_To_Accident"]
    binary_cols = [c for c in binary_cols if c in df.columns]

    st.divider()

    # ---- Sidebar-driven filters ----
    st.sidebar.subheader("Filters")
    race_options = sorted(df["Race"].dropna().unique().tolist())
    selected_races = st.sidebar.multiselect("Filter by Race", race_options, default=race_options)
    df_filtered = df[df["Race"].isin(selected_races)] if selected_races else df

    st.divider()

    # ---- Categorical distribution bar chart ----
    st.subheader("Top Categories Bar Chart")
    cat_col = st.selectbox("Select a categorical column", categorical_columns,
                            index=categorical_columns.index("Race") if "Race" in categorical_columns else 0)
    top_n = st.slider("Number of top categories to show", min_value=3, max_value=20, value=8)
    st.plotly_chart(get_bar_plot(df_filtered, cat_col, top_n), use_container_width=True)

    st.divider()

    # ---- Year distribution histogram ----
    st.subheader("Interactive Histogram")
    hist_column = st.selectbox("Select numeric column for histogram", numeric_columns,
                                index=numeric_columns.index("Year") if "Year" in numeric_columns else 0)
    color_col = st.selectbox("Select grouping color", categorical_columns,
                              index=categorical_columns.index("Violation_Type") if "Violation_Type" in categorical_columns else 0)
    st.plotly_chart(get_hist(df_filtered, hist_column, color_col), use_container_width=True)

    st.divider()

    # ---- Categorical vs categorical grouped bar ----
    st.subheader("Categorical vs. Categorical")
    col1, col2 = st.columns(2)
    with col1:
        x_cat = st.selectbox("X-axis category", categorical_columns,
                              index=categorical_columns.index("Race") if "Race" in categorical_columns else 0, key="x_cat")
    with col2:
        hue_cat = st.selectbox("Group / hue category", categorical_columns,
                                index=categorical_columns.index("Violation_Type") if "Violation_Type" in categorical_columns else 1, key="hue_cat")
    st.plotly_chart(get_stacked_bar(df_filtered, x_cat, hue_cat), use_container_width=True)

    st.divider()

    # ---- Box plot: numeric by categorical ----
    st.subheader("Interactive Box Plot")
    box_x = st.selectbox("Category (X-axis)", categorical_columns,
                          index=categorical_columns.index("Violation_Type") if "Violation_Type" in categorical_columns else 0, key="box_x")
    box_y = st.selectbox("Numeric column (Y-axis)", numeric_columns,
                          index=numeric_columns.index("Year") if "Year" in numeric_columns else 0, key="box_y")
    st.plotly_chart(get_box_plot(df_filtered, box_x, box_y), use_container_width=True)

    st.divider()

    # ---- Pie chart ----
    st.subheader("Composition Pie Chart")
    pie_col = st.selectbox("Select column for pie chart", categorical_columns,
                            index=categorical_columns.index("Gender") if "Gender" in categorical_columns else 0, key="pie_col")
    st.plotly_chart(get_pie_chart(df_filtered, pie_col), use_container_width=True)

    st.divider()

    # ---- Year trend line chart ----
    if "Year" in df.columns:
        st.subheader("Stops by Vehicle Model Year")
        valid_years = df["Year"].dropna().astype(int)
        min_year, max_year = int(valid_years.min()), int(valid_years.max())
        year_range = st.slider("Select year range", min_value=min_year, max_value=max_year,
                                value=(max(min_year, 1990), min(max_year, 2020)))
        st.plotly_chart(get_line_trend(df_filtered, year_range[0], year_range[1]), use_container_width=True)

    st.divider()

    # ---- Correlation heatmap of binary safety flags ----
    if len(binary_cols) >= 2:
        st.subheader("Correlation Between Safety/Accident Flags")
        st.plotly_chart(get_corr_heatmap(df_filtered, binary_cols), use_container_width=True)

    st.divider()

    # ---- Scatter plot (numeric vs numeric, if enough numeric columns exist) ----
    if len(numeric_columns) >= 2:
        st.subheader("Interactive Scatter Plot")
        c1, c2, c3 = st.columns(3)
        with c1:
            x_axis = st.selectbox("Select X-axis", numeric_columns, index=0, key="scatter_x")
        with c2:
            y_axis = st.selectbox("Select Y-axis", numeric_columns,
                                   index=min(1, len(numeric_columns) - 1), key="scatter_y")
        with c3:
            scatter_color = st.selectbox("Select color grouping", categorical_columns, index=0, key="scatter_color")
        st.plotly_chart(get_scatter_plot(df_filtered, x_axis, y_axis, scatter_color), use_container_width=True)
