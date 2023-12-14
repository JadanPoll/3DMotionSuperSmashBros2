from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd


def visualize_dbscan_clusters(ax, features, clusters,pca,scaler, colormap=plt.cm.Spectral):
    """
    Visualizes DBSCAN clusters using PCA.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axes for plotting.
        features (numpy.ndarray): The input features.
        clusters (numpy.ndarray): The cluster labels assigned by DBSCAN.
        pca (PCA, optional): The PCA model for dimensionality reduction. Default is None.
        scaler (Scaler, optional): The feature scaler. Default is None.
        colormap (function, optional): The colormap for cluster visualization. Default is plt.cm.Spectral.
    """
    if scaler:
        features = scaler.transform(features)
    if pca:
        features = pca.transform(features)

    unique_labels = set(clusters)
    colors = [colormap(each) for each in np.linspace(0, 1, len(unique_labels))]

    ax.clear()

    def plot_dbscan_clusters(xy, color, label):
        ax.scatter(xy[:, 0], xy[:, 1], c=[color], s=30, label=label)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Noise points are black
        class_member_mask = (clusters == k)
        xy = features[class_member_mask]
        color = col[:3]
        plot_dbscan_clusters(xy, color, f'Cluster {k}')

    ax.set_title('DBSCAN Clusters')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    plt.draw()  # Update the plot
    plt.pause(0.01)

# Example usage:
# visualize_dbscan_clusters(your_features, your_clusters)

# Function for DBSCAN clustering

dbscan = None
initialized=False
def dbscan_clustering(features, epsilon, min_samples, pca,scaler):
    global dbscan
    global initialized

    if not initialized:
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
        initialized=True
    
    # Ensure that features is a NumPy array or convert DataFrame to NumPy array
    if isinstance(features, pd.DataFrame):
        features = features.values

    data_scaled = scaler.fit_transform(features)
    data_scaled = pca.fit_transform(data_scaled)
    clusters = dbscan.fit_predict(data_scaled)
    return clusters, pca, scaler, dbscan


def dbscan_clustering_predict(features, scaler, pca, dbscan):
    """
    Predict DBSCAN clusters for the given features.

    Parameters:
    - features: Input features (DataFrame or NumPy array).
    - scaler: StandardScaler instance for scaling features.
    - pca: PCA instance for dimensionality reduction.
    - dbscan: DBSCAN instance for clustering.

    Returns:
    - clusters: Predicted clusters.
    """
    if not features.any():
        return np.array([])  # Return empty array for empty features

    if isinstance(features, pd.DataFrame):
        features = features.values

    try:
        data_scaled = scaler.transform(features)
        data_scaled = pca.transform(data_scaled)
        clusters = dbscan.fit_predict(data_scaled)
    except Exception as e:
        print(f"An error occurred during clustering: {e}")
        return np.array([])

    return clusters
