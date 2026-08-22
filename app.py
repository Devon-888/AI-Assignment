import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Clustering Analysis",
    page_icon="🛍️",
    layout="wide"
)

# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


st.sidebar.title("🛍️ Customer Segmentation")




uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# Stop until user uploads CSV
if uploaded_file is None:

    st.info(
        "Please upload a CSV file to start clustering analysis."
    )

    st.stop()


try:

    df = load_data(uploaded_file)

    st.sidebar.success(
        "Dataset loaded successfully."
    )


except Exception as e:

    st.error(
        f"Unable to load dataset.\n\n{e}"
    )

    st.stop()



if df.empty:

    st.error(
        "Dataset is empty."
    )

    st.stop()



if len(df) < 5:

    st.error(
        "Dataset must contain at least 5 records."
    )

    st.stop()



# ----------------------------------------------------------------------------
# Main Title
# ----------------------------------------------------------------------------

st.title(
    "🛍️ Customer Segmentation Clustering Analysis"
)


st.caption(
    "Interactive comparison of DBSCAN / Agglomerative Clustering / Spectral Clustering"
)



# ----------------------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------------------

all_cols = [
    c for c in df.columns
    if c != "CustomerID"
]



default_features = [
    c for c in [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "Gender"
    ]
    if c in df.columns
]



st.sidebar.subheader(
    "Feature Selection"
)



feature_cols_selected = st.sidebar.multiselect(
    "Features used for clustering",
    options=all_cols,
    default=default_features
)



if len(feature_cols_selected) < 2:

    st.warning(
        "Please select at least 2 features for clustering."
    )

    st.stop()


# -----------------------------
# Data Preprocessing
# -----------------------------
# Note: this is an unsupervised clustering task (no target column), so
# preprocessing is fit on the full dataset — there's no train/test split
# to worry about leakage across.

df_proc = df.copy()

X_raw = df_proc[feature_cols_selected].copy()


# ------------------------------------------------------------------------
# Missing Value Handling
# ------------------------------------------------------------------------
# Scaling method (RobustScaler) and outlier capping (IQR) are fixed choices
# rather than user-adjustable options, since they are the recommended
# defaults for this kind of data (income / spending scores are prone to
# outliers). The only thing left adjustable is how missing values are
# imputed, since that's a meaningful methodological choice worth showing.

st.sidebar.subheader("Missing Value Handling")

impute_choice = st.sidebar.selectbox(
    "Method for filling missing values",
    options=[
        "Median (recommended, resistant to outliers)",
        "Mean"
    ],
    index=0,
    help="Median is less affected by extreme values (e.g. a very high "
         "income outlier), so it's the safer default. Mean is provided "
         "as an alternative for comparison."
)

use_mean = impute_choice.startswith("Mean")

HIGH_CARDINALITY_THRESHOLD = 10  # unique values above this -> label encode instead of one-hot


# --- Step 1: Encode categorical columns ---
# Low-cardinality categoricals (e.g. Gender) are one-hot encoded so the model
# doesn't infer a false ordinal relationship between categories.
# High-cardinality categoricals fall back to label encoding to avoid an
# explosion of dummy columns.
cat_cols = [c for c in X_raw.columns if X_raw[c].dtype == "object"]

onehot_cols, label_cols = [], []
for col in cat_cols:
    n_unique = X_raw[col].astype(str).nunique()
    if n_unique <= HIGH_CARDINALITY_THRESHOLD:
        onehot_cols.append(col)
    else:
        label_cols.append(col)

for col in label_cols:
    encoder = LabelEncoder()
    X_raw[col] = encoder.fit_transform(
        X_raw[col].astype(str)
    )

if onehot_cols:
    X_raw = pd.get_dummies(X_raw, columns=onehot_cols, dummy_na=False)
    # get_dummies produces boolean columns; convert to int for downstream math
    bool_cols = X_raw.select_dtypes(include="bool").columns
    X_raw[bool_cols] = X_raw[bool_cols].astype(int)


# --- Step 2: Convert numeric, handle infinities ---
for col in X_raw.columns:

    X_raw[col] = pd.to_numeric(
        X_raw[col],
        errors="coerce"
    )


X_raw = X_raw.replace(
    [np.inf, -np.inf],
    np.nan
)


# --- Step 3: Missing value count (before filling), for transparency in EDA ---
missing_before = X_raw.isna().sum()
missing_before = missing_before[missing_before > 0]


# --- Step 4: Outlier capping (IQR method, always applied) ---
for col in X_raw.columns:
    q1 = X_raw[col].quantile(0.25)
    q3 = X_raw[col].quantile(0.75)
    iqr = q3 - q1
    if pd.notna(iqr) and iqr > 0:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        X_raw[col] = X_raw[col].clip(lower_bound, upper_bound)


# --- Step 5: Fill missing values (Mean or Median, per sidebar choice) ---
# The fill value actually used for each column is recorded here so the EDA
# tab can display it later, even if that column later gets dropped for
# having zero variance (e.g. a column that was entirely missing and got
# filled with a single constant).
fill_values_used = {}

for col in X_raw.columns:

    if X_raw[col].isna().sum() > 0:

        fill_value = (
            X_raw[col].mean()
            if use_mean
            else X_raw[col].median()
        )

        # If entire column is NaN
        if pd.isna(fill_value):

            fill_values_used[col] = 0
            X_raw[col] = X_raw[col].fillna(0)

        else:

            fill_values_used[col] = round(fill_value, 2)
            X_raw[col] = X_raw[col].fillna(fill_value)


# Final safety check
X_raw = X_raw.fillna(0)


# --- Step 6: Drop zero-variance features ---
# A column with a single unique value carries no clustering signal.
zero_var_cols = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
if zero_var_cols:
    X_raw = X_raw.drop(columns=zero_var_cols)
    st.sidebar.caption(
        f"Dropped zero-variance feature(s): {', '.join(zero_var_cols)}"
    )


# --- Step 7: Feature Scaling (RobustScaler, fixed) ---
scaler = RobustScaler()

X_scaled = scaler.fit_transform(
    X_raw
)

# ----------------------------------------------------------------------------
# PCA Dimension Reduction
# ----------------------------------------------------------------------------

try:

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

except Exception as e:

    st.error(
        f"PCA Error:\n{e}"
    )

    st.stop()

# ----------------------------------------------------------------------------
# Scatter Plot Selection
# ----------------------------------------------------------------------------

scatter_x_default = (
    "Annual Income (k$)"
    if "Annual Income (k$)" in df.columns
    else all_cols[0]
)



scatter_y_default = (
    "Spending Score (1-100)"
    if "Spending Score (1-100)" in df.columns
    else all_cols[-1]
)



col1, col2 = st.sidebar.columns(2)



scatter_x = col1.selectbox(
    "Scatter Plot X Axis",
    options=all_cols,
    index=all_cols.index(
        scatter_x_default
    )
)



scatter_y = col2.selectbox(
    "Scatter Plot Y Axis",
    options=all_cols,
    index=all_cols.index(
        scatter_y_default
    )
)




# ----------------------------------------------------------------------------
# Initialize Results
# ----------------------------------------------------------------------------

dbscan_silhouette = np.nan
dbscan_db_score = np.nan

agg_silhouette = np.nan
agg_db = np.nan

spec_silhouette = np.nan
spec_db = np.nan


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_eda, tab_dbscan, tab_agg, tab_spec, tab_compare = st.tabs(
[
"📊 Data Exploration",
"🔵 DBSCAN",
"🌳 Agglomerative Clustering",
"🕸️ Spectral Clustering",
"⚖️ Method Comparison"
]
)



# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------

with tab_eda:

    st.subheader(
        "Data Preview"
    )


    st.dataframe(
        df.head(10),
        use_container_width=True
    )



    c1, c2, c3 = st.columns(3)



    c1.metric(
        " of Samples",
        df.shape[0]
    )


    c2.metric(
        "Number of Features",
        df.shape[1] - 1
    )


    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )



    st.subheader(
        "Missing Value Handling"
    )

    if len(missing_before) > 0:

        st.write(
            f"The following selected features had missing values, filled "
            f"using **{'Mean' if use_mean else 'Median'}** "
            f"(after outlier capping):"
        )

        missing_table = pd.DataFrame({
            "Feature": missing_before.index,
            "Missing Count": missing_before.values,
            "Fill Value Used": [
                fill_values_used.get(col, "N/A")
                for col in missing_before.index
            ]
        })

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    else:

        st.write(
            "No missing values found in the selected features."
        )



    st.subheader(
        "Numerical Feature Distribution"
    )



    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()



    numeric_cols = [
        c for c in numeric_cols
        if c != "CustomerID"
    ]



    if numeric_cols:

        fig, axes = plt.subplots(
            1,
            len(numeric_cols),
            figsize=(5 * len(numeric_cols), 4)
        )



        if len(numeric_cols) == 1:
            axes = [axes]



        for ax, col in zip(
            axes,
            numeric_cols
        ):

            sns.histplot(
                df[col],
                kde=True,
                ax=ax
            )

            ax.set_title(
                col
            )



        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)



    if "Gender" in df.columns:

        st.subheader(
            "Pairwise Relationship by Gender"
        )



        pair_cols = numeric_cols



        g = sns.pairplot(
            df[pair_cols + ["Gender"]],
            hue="Gender"
        )


        st.pyplot(
            g.fig
        )


        plt.close(
            g.fig
        )

# ----------------------------------------------------------------------------
# DBSCAN Clustering
# ----------------------------------------------------------------------------

with tab_dbscan:

    st.subheader(
        "DBSCAN Clustering"
    )

    st.markdown(
        """
        DBSCAN is a density-based clustering algorithm.
        It identifies clusters based on data density
        and can detect noise points.

        The system automatically searches for suitable
        min_samples and eps values based on clustering
        evaluation metrics.
        """
    )


    # ------------------------------------------------------------------------
    # Automatic DBSCAN Parameter Search
    # ------------------------------------------------------------------------

    best_score = -np.inf
    best_eps = None
    best_min_samples = None
    best_labels = None
    best_silhouette = None
    best_db = None
    best_clusters = None
    best_noise = None


    # Test different min_samples values
    min_samples_range = range(
        3,
        11
    )


    # Test different eps values
    eps_range = np.arange(
        0.2,
        3.01,
        0.05
    )


    for test_min_samples in min_samples_range:

        for test_eps in eps_range:

            temp_model = DBSCAN(
                eps=test_eps,
                min_samples=test_min_samples
            )


            temp_labels = temp_model.fit_predict(
                X_scaled
            )


            # ------------------------------------------------------------
            # Count clusters
            # ------------------------------------------------------------

            unique_labels = set(
                temp_labels
            )


            n_clusters = len(
                unique_labels
            ) - (
                1 if -1 in unique_labels else 0
            )


            # Need at least 2 clusters
            if n_clusters < 2:
                continue


            # ------------------------------------------------------------
            # Calculate noise percentage
            # ------------------------------------------------------------

            noise_count = np.sum(
                temp_labels == -1
            )


            noise_percentage = (
                noise_count /
                len(temp_labels)
            ) * 100


            # Avoid solutions with excessive noise
            if noise_percentage > 30:
                continue


            # ------------------------------------------------------------
            # Remove noise for evaluation
            # ------------------------------------------------------------

            mask = (
                temp_labels != -1
            )


            X_clustered = X_scaled[
                mask
            ]


            labels_clustered = temp_labels[
                mask
            ]


            # Need at least 2 valid clusters
            if len(
                np.unique(labels_clustered)
            ) < 2:
                continue


            # ------------------------------------------------------------
            # Evaluation metrics
            # ------------------------------------------------------------

            temp_silhouette = silhouette_score(
                X_clustered,
                labels_clustered
            )


            temp_db = davies_bouldin_score(
                X_clustered,
                labels_clustered
            )


            # ------------------------------------------------------------
            # Combined score
            #
            # Silhouette:
            # higher = better
            #
            # DBI:
            # lower = better
            # ------------------------------------------------------------

            combined_score = (
                temp_silhouette -
                0.1 * temp_db
            )


            # ------------------------------------------------------------
            # Save best result
            # ------------------------------------------------------------

            if combined_score > best_score:

                best_score = combined_score

                best_eps = test_eps

                best_min_samples = (
                    test_min_samples
                )

                best_labels = temp_labels

                best_silhouette = (
                    temp_silhouette
                )

                best_db = temp_db

                best_clusters = n_clusters

                best_noise = noise_count


    # ------------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------------

    if best_labels is None:

        st.error(
            "No suitable DBSCAN parameters were found."
        )

        st.stop()


    dbscan_labels = best_labels
    
    dbscan_silhouette = best_silhouette
    dbscan_db_score = best_db
    # ------------------------------------------------------------------------
    # Display Automatically Selected Parameters
    # ------------------------------------------------------------------------

    st.success(
        "DBSCAN parameters were automatically selected."
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "min_samples",
        best_min_samples
    )


    c2.metric(
        "eps",
        f"{best_eps:.2f}"
    )


    c3.metric(
        "Number of Clusters",
        best_clusters
    )


    c4.metric(
        "Noise Points",
        best_noise
    )


    # ------------------------------------------------------------------------
    # Evaluation Metrics
    # ------------------------------------------------------------------------

    m1, m2 = st.columns(2)


    m1.metric(
        "Silhouette Score",
        f"{best_silhouette:.3f}"
    )


    m2.metric(
        "Davies-Bouldin Index",
        f"{best_db:.3f}"
    )


    # ------------------------------------------------------------------------
    # k-distance Graph
    # ------------------------------------------------------------------------

    with st.expander(
        "View k-distance Graph"
    ):

        neighbors = NearestNeighbors(
            n_neighbors=best_min_samples
        )


        distances, _ = neighbors.fit(
            X_scaled
        ).kneighbors(
            X_scaled
        )


        k_distances = np.sort(
            distances[:, -1]
        )


        fig, ax = plt.subplots(
            figsize=(8, 3.5)
        )


        ax.plot(
            k_distances
        )


        ax.axhline(
            best_eps,
            linestyle="--",
            label=f"Selected eps = {best_eps:.2f}"
        )


        ax.set_title(
            f"k-distance Graph (k={best_min_samples})"
        )


        ax.set_xlabel(
            "Points"
        )


        ax.set_ylabel(
            "Distance"
        )


        ax.legend()


        st.pyplot(fig)

        plt.close(fig)


    # ------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------

    p1, p2 = st.columns(2)


    with p1:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        sc = ax.scatter(
            df[scatter_x],
            df[scatter_y],
            c=dbscan_labels,
            cmap="tab10",
            s=45
        )


        ax.set_title(
            "DBSCAN Clustering"
        )


        ax.set_xlabel(
            scatter_x
        )


        ax.set_ylabel(
            scatter_y
        )


        plt.colorbar(
            sc,
            ax=ax,
            label="Cluster (-1 = Noise)"
        )


        st.pyplot(fig)

        plt.close(fig)


    with p2:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        sc = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=dbscan_labels,
            cmap="tab10",
            s=45
        )


        ax.set_title(
            "DBSCAN - PCA Projection"
        )


        ax.set_xlabel(
            "PCA 1"
        )


        ax.set_ylabel(
            "PCA 2"
        )


        plt.colorbar(
            sc,
            ax=ax,
            label="Cluster"
        )


        st.pyplot(fig)

        plt.close(fig)

# ----------------------------------------------------------------------------
# Agglomerative Clustering
# ----------------------------------------------------------------------------

with tab_agg:


    st.subheader(
        "Agglomerative Clustering (Ward Linkage)"
    )



    with st.expander(
        "View Dendrogram"
    ):


        linked = linkage(
            X_scaled,
            method="ward"
        )


        fig, ax = plt.subplots(
            figsize=(10, 4)
        )


        dendrogram(
            linked,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            ax=ax
        )


        ax.set_title(
            "Dendrogram (Ward Linkage)"
        )


        st.pyplot(fig)

        plt.close(fig)




    with st.expander(
        "View Silhouette Score for Different k Values"
    ):


        agg_sil_scores = []


        K_range = range(
            2,
            11
        )


        for k in K_range:


            agg_tmp = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward"
            )


            labels_tmp = agg_tmp.fit_predict(
                X_scaled
            )


            agg_sil_scores.append(
                silhouette_score(
                    X_scaled,
                    labels_tmp
                )
            )



        fig, ax = plt.subplots(
            figsize=(7, 3.5)
        )


        ax.plot(
            list(K_range),
            agg_sil_scores,
            marker="o"
        )


        ax.set_title(
            "Silhouette Score vs Number of Clusters"
        )


        ax.set_xlabel(
            "Number of Clusters (k)"
        )


        ax.set_ylabel(
            "Silhouette Score"
        )


        st.pyplot(fig)

        plt.close(fig)




    k_agg = st.slider(
        "Number of Clusters (k)",
        min_value=2,
        max_value=10,
        value=5,
        key="agg_k"
    )



    agg = AgglomerativeClustering(
        n_clusters=k_agg,
        linkage="ward"
    )



    agg_labels = agg.fit_predict(
        X_scaled
    )



    agg_silhouette = silhouette_score(
        X_scaled,
        agg_labels
    )


    agg_db = davies_bouldin_score(
        X_scaled,
        agg_labels
    )



    m1, m2 = st.columns(2)



    m1.metric(
        "Silhouette Score",
        f"{agg_silhouette:.3f}"
    )


    m2.metric(
        "Davies-Bouldin Index",
        f"{agg_db:.3f}"
    )



    p1, p2 = st.columns(2)



    with p1:


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        sc = ax.scatter(
            df[scatter_x],
            df[scatter_y],
            c=agg_labels,
            cmap="tab10",
            s=45
        )


        ax.set_title(
            f"Agglomerative Clustering (k={k_agg})"
        )


        ax.set_xlabel(
            scatter_x
        )


        ax.set_ylabel(
            scatter_y
        )


        plt.colorbar(
            sc,
            ax=ax,
            label="Cluster"
        )


        st.pyplot(fig)

        plt.close(fig)



    with p2:


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        sc = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=agg_labels,
            cmap="tab10",
            s=45
        )


        ax.set_title(
            "Agglomerative Clustering - PCA Projection"
        )


        plt.colorbar(
            sc,
            ax=ax,
            label="Cluster"
        )


        st.pyplot(fig)

        plt.close(fig)

# ----------------------------------------------------------------------------
# Spectral Clustering
# ----------------------------------------------------------------------------

with tab_spec:

    st.subheader(
        "Spectral Clustering"
    )

    n_clusters_spec = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5,
        key="spec_k"
    )


    spectral = SpectralClustering(
        n_clusters=n_clusters_spec,
        affinity="nearest_neighbors",
        random_state=42
    )


    spec_labels = spectral.fit_predict(
        X_scaled
    )


    spec_silhouette = silhouette_score(
        X_scaled,
        spec_labels
    )


    spec_db = davies_bouldin_score(
        X_scaled,
        spec_labels
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Silhouette Score",
        f"{spec_silhouette:.3f}"
    )


    c2.metric(
        "Davies-Bouldin Index",
        f"{spec_db:.3f}"
    )


    fig, ax = plt.subplots(
        figsize=(6,5)
    )


    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=spec_labels,
        cmap="tab10",
        s=45
    )


    ax.set_title(
        "Spectral Clustering - PCA Projection"
    )


    plt.colorbar(
        scatter,
        ax=ax
    )


    st.pyplot(fig)

    plt.close(fig)

# ----------------------------------------------------------------------------
# Method Comparison
# ----------------------------------------------------------------------------

with tab_compare:

    st.subheader(
        "Clustering Method Comparison"
    )

    # ------------------------------------------------------------
    # Calculate number of clusters
    # ------------------------------------------------------------

    dbscan_n_clusters = len(
        set(dbscan_labels) - {-1}
    )

    agg_n_clusters = len(
        set(agg_labels)
    )

    spec_n_clusters = len(
        set(spec_labels)
    )

    # ------------------------------------------------------------
    # Comparison DataFrame
    # ------------------------------------------------------------

    comparison = pd.DataFrame({

        "Method": [
            "DBSCAN",
            "Agglomerative",
            "Spectral"
        ],

        "Number of Clusters": [
            dbscan_n_clusters,
            agg_n_clusters,
            spec_n_clusters
        ],

        "Silhouette Score": [
            dbscan_silhouette,
            agg_silhouette,
            spec_silhouette
        ],

        "Davies-Bouldin Index": [
            dbscan_db_score,
            agg_db,
            spec_db
        ]

    })

    # ------------------------------------------------------------
    # Display table
    # ------------------------------------------------------------

    display_comparison = comparison.copy()

    display_comparison[
        "Silhouette Score"
    ] = display_comparison[
        "Silhouette Score"
    ].apply(
        lambda x:
        "N/A"
        if pd.isna(x)
        else round(x, 3)
    )

    display_comparison[
        "Davies-Bouldin Index"
    ] = display_comparison[
        "Davies-Bouldin Index"
    ].apply(
        lambda x:
        "N/A"
        if pd.isna(x)
        else round(x, 3)
    )

    st.dataframe(
        display_comparison,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Silhouette Score Chart
    # ------------------------------------------------------------

    chart_data = comparison.dropna(
        subset=["Silhouette Score"]
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        chart_data["Method"],
        chart_data["Silhouette Score"]
    )

    ax.set_ylabel(
        "Silhouette Score"
    )

    ax.set_title(
        "Silhouette Score Comparison"
    )

    st.pyplot(fig)

    plt.close(fig)
