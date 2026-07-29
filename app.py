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

st.set_page_config(page_title="顾客细分聚类分析", page_icon="🛍️", layout="wide")

# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.sidebar.title("🛍️ 顾客细分")
st.sidebar.markdown("基于 Mall Customers 数据集的聚类分析")

uploaded_file = st.sidebar.file_uploader("上传 CSV 文件（可选，默认使用内置数据）", type=["csv"])

default_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data(default_path)

st.title("🛍️ 顾客细分聚类分析")
st.caption("DBSCAN / 层次聚类（Agglomerative）/ 谱聚类（Spectral Clustering）交互式对比")

# ----------------------------------------------------------------------------
# 特征选择
# ----------------------------------------------------------------------------
all_cols = [c for c in df.columns if c != "CustomerID"]
default_features = [c for c in ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"] if c in df.columns]

st.sidebar.subheader("特征选择")
feature_cols_selected = st.sidebar.multiselect(
    "用于聚类的特征", options=all_cols, default=default_features
)

if len(feature_cols_selected) < 2:
    st.warning("请至少选择 2 个特征用于聚类。")
    st.stop()

# ----------------------------------------------------------------------------
# 预处理
# ----------------------------------------------------------------------------
df_proc = df.copy()
feature_cols = []

for col in feature_cols_selected:
    if df_proc[col].dtype == object:
        le = LabelEncoder()
        df_proc[col + "_enc"] = le.fit_transform(df_proc[col])
        feature_cols.append(col + "_enc")
    else:
        feature_cols.append(col)


# 修改这里
X_raw = df_proc[feature_cols]

st.write(X_raw.head())
st.write(X_raw.dtypes)
st.write(X_raw.isnull().sum())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 用于二维散点图的两个轴（默认用 Annual Income / Spending Score，如果存在）
scatter_x_default = "Annual Income (k$)" if "Annual Income (k$)" in df.columns else all_cols[0]
scatter_y_default = "Spending Score (1-100)" if "Spending Score (1-100)" in df.columns else all_cols[-1]

col1, col2 = st.sidebar.columns(2)
scatter_x = col1.selectbox("散点图 X 轴", options=all_cols, index=all_cols.index(scatter_x_default))
scatter_y = col2.selectbox("散点图 Y 轴", options=all_cols, index=all_cols.index(scatter_y_default))

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_eda, tab_dbscan, tab_agg, tab_spec, tab_compare = st.tabs(
    ["📊 数据探索", "🔵 DBSCAN", "🌳 层次聚类", "🕸️ 谱聚类", "⚖️ 三种方法对比"]
)

# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", df.shape[0])
    c2.metric("特征数", df.shape[1] - 1)
    c3.metric("缺失值", int(df.isnull().sum().sum()))

    st.subheader("数值特征分布")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "CustomerID"]
    if numeric_cols:
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "Gender" in df.columns:
        st.subheader("按性别分组的成对关系图")
        pair_cols = [c for c in numeric_cols]
        g = sns.pairplot(df[pair_cols + ["Gender"]], hue="Gender")
        st.pyplot(g.fig)
        plt.close(g.fig)

# ----------------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------------
with tab_dbscan:
    st.subheader("DBSCAN 聚类")
    st.markdown("基于密度的聚类方法，能自动识别噪声点（离群点）。")

    dc1, dc2 = st.columns(2)
    min_samples = dc1.slider("min_samples", min_value=2, max_value=20, value=5, key="db_min_samples")
    eps = dc2.slider("eps（邻域半径）", min_value=0.1, max_value=3.0, value=0.8, step=0.05, key="db_eps")

    with st.expander("查看 k-距离图（用于辅助选择 eps）"):
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        distances, _ = neighbors.fit(X_scaled).kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(k_distances)
        ax.set_title(f"k-distance graph (k={min_samples})")
        ax.set_xlabel("Points sorted by distance")
        ax.set_ylabel(f"Distance to {min_samples}-th NN")
        st.pyplot(fig)
        plt.close(fig)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = int(list(dbscan_labels).count(-1))

    m1, m2, m3 = st.columns(3)
    m1.metric("聚类数", n_clusters_db)
    m2.metric("噪声点数", n_noise, f"{n_noise/len(dbscan_labels):.1%}")

    if n_clusters_db >= 2:
        mask = dbscan_labels != -1
        dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        dbscan_db_score = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
        m3.metric("轮廓系数", f"{dbscan_silhouette:.3f}")
        st.caption(f"Davies-Bouldin 指数: {dbscan_db_score:.3f}")
    else:
        dbscan_silhouette, dbscan_db_score = np.nan, np.nan
        st.warning("聚类数不足 2，无法计算轮廓系数，请调整参数。")

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(df[scatter_x], df[scatter_y], c=dbscan_labels, cmap="tab10", s=45)
        ax.set_title(f"DBSCAN (eps={eps}, min_samples={min_samples})")
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        plt.colorbar(sc, ax=ax, label="Cluster (-1=noise)")
        st.pyplot(fig)
        plt.close(fig)
    with p2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="tab10", s=45)
        ax.set_title("DBSCAN - PCA 投影")
        plt.colorbar(sc, ax=ax, label="Cluster")
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------
# Agglomerative
# ----------------------------------------------------------------------------
with tab_agg:
    st.subheader("层次聚类（Agglomerative, Ward Linkage）")

    with st.expander("查看谱系图（Dendrogram）"):
        linked = linkage(X_scaled, method="ward")
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90.0, ax=ax)
        ax.set_title("Dendrogram (Ward linkage)")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("查看不同 k 值下的轮廓系数"):
        agg_sil_scores = []
        K_range = range(2, 11)
        for k in K_range:
            agg_tmp = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels_tmp = agg_tmp.fit_predict(X_scaled)
            agg_sil_scores.append(silhouette_score(X_scaled, labels_tmp))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(list(K_range), agg_sil_scores, "-o", color="seagreen")
        ax.set_title("Silhouette Score vs Number of Clusters")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette Score")
        st.pyplot(fig)
        plt.close(fig)

    k_agg = st.slider("聚类数 k", min_value=2, max_value=10, value=5, key="agg_k")

    agg = AgglomerativeClustering(n_clusters=k_agg, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)

    agg_silhouette = silhouette_score(X_scaled, agg_labels)
    agg_db = davies_bouldin_score(X_scaled, agg_labels)

    m1, m2 = st.columns(2)
    m1.metric("轮廓系数", f"{agg_silhouette:.3f}")
    m2.metric("Davies-Bouldin 指数", f"{agg_db:.3f}")

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(df[scatter_x], df[scatter_y], c=agg_labels, cmap="tab10", s=45)
        ax.set_title(f"Agglomerative Clustering (k={k_agg}, ward)")
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        plt.colorbar(sc, ax=ax, label="Cluster")
        st.pyplot(fig)
        plt.close(fig)
    with p2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap="tab10", s=45)
        ax.set_title("Agglomerative Clustering - PCA 投影")
        plt.colorbar(sc, ax=ax, label="Cluster")
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------
# Spectral
# ----------------------------------------------------------------------------
with tab_spec:
    st.subheader("谱聚类（Spectral Clustering）")

    with st.expander("查看不同 k 值下的轮廓系数（计算较慢，可能需要几秒）"):
        if st.checkbox("运行 k=2~10 的轮廓系数扫描", key="spec_scan"):
            spec_sil_scores = []
            K_range = range(2, 11)
            for k in K_range:
                spec_tmp = SpectralClustering(
                    n_clusters=k, affinity="nearest_neighbors", n_neighbors=10,
                    random_state=42, assign_labels="kmeans"
                )
                labels_tmp = spec_tmp.fit_predict(X_scaled)
                spec_sil_scores.append(silhouette_score(X_scaled, labels_tmp))
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(list(K_range), spec_sil_scores, "-o", color="indianred")
            ax.set_title("Silhouette Score vs Number of Clusters")
            ax.set_xlabel("k")
            ax.set_ylabel("Silhouette Score")
            st.pyplot(fig)
            plt.close(fig)

    sc1, sc2 = st.columns(2)
    k_spec = sc1.slider("聚类数 k", min_value=2, max_value=10, value=5, key="spec_k")
    n_neighbors_spec = sc2.slider("n_neighbors（近邻数）", min_value=3, max_value=30, value=10, key="spec_nn")

    spectral = SpectralClustering(
        n_clusters=k_spec, affinity="nearest_neighbors", n_neighbors=n_neighbors_spec,
        random_state=42, assign_labels="kmeans"
    )
    spectral_labels = spectral.fit_predict(X_scaled)

    spectral_silhouette = silhouette_score(X_scaled, spectral_labels)
    spectral_db = davies_bouldin_score(X_scaled, spectral_labels)

    m1, m2 = st.columns(2)
    m1.metric("轮廓系数", f"{spectral_silhouette:.3f}")
    m2.metric("Davies-Bouldin 指数", f"{spectral_db:.3f}")

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(df[scatter_x], df[scatter_y], c=spectral_labels, cmap="tab10", s=45)
        ax.set_title(f"Spectral Clustering (k={k_spec})")
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        plt.colorbar(sc, ax=ax, label="Cluster")
        st.pyplot(fig)
        plt.close(fig)
    with p2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=spectral_labels, cmap="tab10", s=45)
        ax.set_title("Spectral Clustering - PCA 投影")
        plt.colorbar(sc, ax=ax, label="Cluster")
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("三种聚类方法对比")
    st.markdown("以下对比使用各标签页中当前设置的参数。")

    comparison = pd.DataFrame({
        "Method": ["DBSCAN", "Agglomerative", "Spectral"],
        "Num_Clusters": [n_clusters_db, k_agg, k_spec],
        "Silhouette_Score": [dbscan_silhouette, agg_silhouette, spectral_silhouette],
        "Davies_Bouldin": [dbscan_db_score, agg_db, spectral_db],
        "Noise_Points": [n_noise, 0, 0],
    })
    st.dataframe(comparison, use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels_dict = {"DBSCAN": dbscan_labels, "Agglomerative": agg_labels, "Spectral": spectral_labels}
    for ax, (name, labels) in zip(axes, labels_dict.items()):
        sc = ax.scatter(df[scatter_x], df[scatter_y], c=labels, cmap="tab10", s=40)
        ax.set_title(name)
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    valid = comparison.dropna(subset=["Silhouette_Score"])
    ax.bar(valid["Method"], valid["Silhouette_Score"], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("轮廓系数对比（越高越好）")
    ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "下载带聚类标签的完整数据 (CSV)",
        data=df.assign(
            DBSCAN_Cluster=dbscan_labels,
            Agglomerative_Cluster=agg_labels,
            Spectral_Cluster=spectral_labels,
        ).to_csv(index=False).encode("utf-8-sig"),
        file_name="customer_segmentation_results.csv",
        mime="text/csv",
    )

st.sidebar.markdown("---")
st.sidebar.caption("由 Customer_Segmentation_Clustering.ipynb 改编为 Streamlit 应用")
