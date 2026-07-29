import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
st.sidebar.markdown(
    "Clustering analysis based on Mall Customers dataset"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (Optional)",
    type=["csv"]
)

default_path = os.path.join(
    os.path.dirname(__file__),
    "Mall_Customers.csv"
)

try:

    if uploaded_file is not None:

        df = load_data(uploaded_file)

        st.sidebar.success("Custom dataset loaded successfully.")

    elif os.path.exists(default_path):

        df = load_data(default_path)

        st.sidebar.info("Using default Mall_Customers.csv")

    else:

        st.error("Default dataset not found.")

        st.stop()

except Exception as e:

    st.error(f"Unable to load dataset.\n\n{e}")

    st.stop()

if df.empty:

    st.error("Dataset is empty.")

    st.stop()

if len(df) < 5:

    st.error("Dataset must contain at least 5 records.")

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

df_proc = df.copy()

X_raw = df_proc[feature_cols_selected].copy()


# Convert categorical columns
for col in X_raw.columns:

    if X_raw[col].dtype == "object":

        encoder = LabelEncoder()

        X_raw[col] = encoder.fit_transform(
            X_raw[col].astype(str)
        )


# Convert all columns to numeric
for col in X_raw.columns:

    X_raw[col] = pd.to_numeric(
        X_raw[col],
        errors="coerce"
    )


# Replace infinite values
X_raw = X_raw.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing values
X_raw = X_raw.fillna(
    X_raw.mean()
)


# Remove remaining missing rows
X_raw = X_raw.dropna()

df = df.loc[X_raw.index]


# Reset index
X_raw = X_raw.reset_index(drop=True)


# Debug
st.write("After preprocessing:")
st.dataframe(X_raw.head())

st.write("Data types:")
st.write(X_raw.dtypes)

st.write("Missing:")
st.write(X_raw.isnull().sum())


# ----------------------------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------------------------
scaler = StandardScaler()


# DEBUG
st.write("===== DEBUG BEFORE SCALING =====")

st.write("X_raw shape:")
st.write(X_raw.shape)

st.write("X_raw preview:")
st.dataframe(X_raw.head())


st.write("Data types:")
st.write(X_raw.dtypes)


st.write("NaN count:")
st.write(X_raw.isna().sum())


st.write("Infinite count:")
st.write(np.isinf(X_raw).sum())


if X_raw.empty:
    st.error("X_raw is empty after preprocessing.")
    st.stop()


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

    st.error(f"PCA Error:\n{e}")

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
        "Number of Samples",
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
        It can automatically identify clusters and detect noise points (outliers).
        """
    )



    dc1, dc2 = st.columns(2)



    min_samples = dc1.slider(
        "min_samples",
        min_value=2,
        max_value=20,
        value=5,
        key="db_min_samples"
    )



    eps = dc2.slider(
        "eps (Neighborhood Radius)",
        min_value=0.1,
        max_value=3.0,
        value=0.8,
        step=0.05,
        key="db_eps"
    )



    with st.expander(
        "View k-distance Graph (for selecting eps)"
    ):


        neighbors = NearestNeighbors(
            n_neighbors=min_samples
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


        ax.set_title(
            f"k-distance graph (k={min_samples})"
        )


        ax.set_xlabel(
            "Points sorted by distance"
        )


        ax.set_ylabel(
            f"Distance to {min_samples}-th NN"
        )


        st.pyplot(fig)

        plt.close(fig)



    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )


try:

    dbscan_labels = dbscan.fit_predict(X_scaled)

except Exception as e:

    st.error(f"DBSCAN Error:\n{e}")

    st.stop()



    n_clusters_db = len(
        set(dbscan_labels)
    ) - (
        1 if -1 in dbscan_labels else 0
    )


    n_noise = int(
        list(dbscan_labels).count(-1)
    )



    m1, m2, m3 = st.columns(3)



    m1.metric(
        "Number of Clusters",
        n_clusters_db
    )


    m2.metric(
        "Noise Points",
        n_noise,
        f"{n_noise / len(dbscan_labels):.1%}"
    )



    if n_clusters_db >= 2:


        mask = dbscan_labels != -1


        dbscan_silhouette = silhouette_score(
            X_scaled[mask],
            dbscan_labels[mask]
        )


        dbscan_db_score = davies_bouldin_score(
            X_scaled[mask],
            dbscan_labels[mask]
        )


        m3.metric(
            "Silhouette Score",
            f"{dbscan_silhouette:.3f}"
        )


        st.caption(
            f"Davies-Bouldin Index: {dbscan_db_score:.3f}"
        )


    else:

        dbscan_silhouette = np.nan

        dbscan_db_score = np.nan


        st.warning(
            "Not enough clusters to calculate evaluation scores. Please adjust parameters."
        )



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
            f"DBSCAN (eps={eps}, min_samples={min_samples})"
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
