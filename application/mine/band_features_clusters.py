import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os

import sys
sys.path.append('.')
from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir

# Define constants and paths
K_RANGE = list(range(2, 6))  # Range of K to test for optimal clustering
K = 3
FIG_SIZE = (8, 10)
META_FILE = '/well/rittscher/users/qdv200/MPN/xenium/data/xenium_keys.csv'
PATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_S100'
OUTPATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_S100_Clusters'

# Function to create directories if they don't exist
def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# Function to determine the optimal number of clusters using the Silhouette method
def determine_optimal_clusters(data, features_col):
    silhouette_scores = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data[features_col])
        silhouette_avg = silhouette_score(data[features_col], labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot the Silhouette method graph
    plt.figure(figsize=FIG_SIZE)
    plt.plot(K_RANGE, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal K')
    plt.savefig(f'{PLOTS_DIR}/silhouette_method.png')

    # Select the optimal number of clusters based on the highest Silhouette score
    optimal_k = K_RANGE[np.argmax(silhouette_scores)]
    return optimal_k

# Function to perform KMeans clustering and generate plots
def perform_clustering_and_generate_plots(data, features_col, cell_type,celltypes_selected=None):
    optimal_k = K#determine_optimal_clusters(data, features_col)
    if cell_type in celltypes_selected:
        optimal_k = celltypes_selected[cell_type]
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[features_col])
    
    # Calculate the mean feature values for each cluster
    cluster_means = data[features_col + ['cluster']].groupby('cluster').mean() + 1e-6
    reorder = sorted(cluster_means.columns)
    cluster_means = cluster_means.loc[:,reorder]
    # Visualize the feature distribution across clusters using a heatmap
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(cluster_means.T, cmap="YlGnBu", annot=True, vmin=0, vmax=1)
    plt.title('Mean Microenvironmental Features per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{cell_type}_abundance.png')

    # Synthetic reference cluster and log2 fold change calculation
    synthetic_reference = pd.Series(1 / len(cluster_means.columns), index=cluster_means.columns)
    cluster_means.loc['synthetic_reference'] = synthetic_reference
    log2_fold_change = np.log2(cluster_means / cluster_means.loc['synthetic_reference'])
    log2_fold_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    log2_fold_change = log2_fold_change.drop('synthetic_reference')

    # Visualize the log2 fold change using a heatmap
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(log2_fold_change.T, cmap="RdBu_r", annot=True, center=0)
    plt.title('Log2 Fold Change of Microenvironmental Features per Cluster (Relative to Synthetic Reference)')
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{cell_type}_abundance_log_fold.png')

    # Save clustered data to CSV
    data.to_csv(f'{CSV_DIR}/{cell_type}.csv')

# Main script execution
if __name__ == "__main__":

    celltypes_selected = {
        'Immune': 3, 'Endothelial': 3, 'Mesenchymal': 3,
        'Epithelial': 3
    }

    for cell_type in celltypes_selected:
        if cell_type in ['Granulocyte/mast']:
            cell_type = 'Granulocyte_mast'
        INPUT_FILE = f'{PATH}/band_features_{cell_type}_cell_id_br2.csv'
        OUT_PATH = f'{OUTPATH}/br2_k{K}/{cell_type}'
        #OUT_PATH = f'{OUTPATH}/br2_custom/{cell_type}'
        mkdirs(OUT_PATH)
        PLOTS_DIR = f'{OUT_PATH}/plots'
        CSV_DIR = f'{OUT_PATH}/csv'
        mkdirs(PLOTS_DIR)
        mkdirs(CSV_DIR)
        print(f'Processing cell type {cell_type}')
        data = pd.read_csv(INPUT_FILE)
        # idx = np.random.randint(0,len(data),500)
        # data = data.iloc[idx,:]
        features_col = list(set(data.columns) - {'sample_id', 'label','cell_id'})
        perform_clustering_and_generate_plots(data, features_col, cell_type,celltypes_selected = celltypes_selected)
