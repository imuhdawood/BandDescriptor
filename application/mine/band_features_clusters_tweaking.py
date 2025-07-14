import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import mixture
from shutil import copy

import sys
sys.path.append('.')
from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir
ANNOT_LEVEL = 'final_lineage'
# Define constants and paths
K_RANGE = [1, 2, 3, 4, 5, 6]#, 13, 15, 17, 19]#list(range(2, 6))  # Range of K to test for optimal clustering
K = 3
FIG_SIZE = (8, 10)

PATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull'
OUTPATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_Clusters'


# Function to create directories if they don't exist
def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# bics
# [-5421401.8405116275, -6853915.775527398, -7225679.659713283, -7447671.106310615, -7542925.863725335, -7868986.030059874]
# aics
# [-5423780.627228234, -6858683.691511552, -7232836.704964985, -7457217.2808298655, -7554861.167512133, -7883310.46311422]
def determine_optimal_clusters_gaussian(data_scaled):
    data_scaled = StandardScaler().fit_transform(data_scaled)
    gmm = mixture.BayesianGaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(data_scaled)
    embed_2d = PCA().fit_transform(data_scaled)
    plt.scatter(x = embed_2d[:,0], y= embed_2d[:,1], c = labels, s=1)
    plt.savefig('/well/rittscher/users/qwi813/CellMiroEnviron/OUTPUT/test.png')
    print()
    #Selecting set of Principal Components
    # n_components = (PCA().fit(data_scaled).explained_variance_ratio_.cumsum()>0.95).argmax()+1
    # data_scaled = PCA(n_components=n_components).fit_transform(data_scaled)
    bics = []
    aics = []
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data_scaled)
        bics.append(gmm.bic(data_scaled))
        aics.append(gmm.aic(data_scaled))
    optimal_k = K_RANGE[np.argmin(bics)]
    return optimal_k

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
    #optimal_k = determine_optimal_clusters_gaussian(data[features_col])#K#determine_optimal_clusters(data, features_col)
    # if cell_type in celltypes_selected:
    #     optimal_k = celltypes_selected[cell_type]
    #kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    # kmeans = mixture.BayesianGaussianMixture(
    # n_components=10,  # Start with a larger number of components
    # weight_concentration_prior_type='dirichlet_process',  # Flexible number of clusters
    # weight_concentration_prior=1e-2,  # Encourage sparsity
    # covariance_type='full',  # Capture complex cluster shapes
    # mean_precision_prior=1e-2,  # Moderate prior on mean precision
    # mean_prior=None,  # Use data mean as prior
    # degrees_of_freedom_prior=None,  # Default to number of features
    # covariance_prior=None,  # Use empirical covariance
    # tol=1e-3,  # Convergence threshold
    # max_iter=300,  # Maximum number of iterations
    # n_init=5,  # Multiple initializations
    # init_params='k-means++',  # Robust initialization
    # random_state=42,  # Reproducibility
    # verbose=1  # Monitor fitting process
    # )


    kmeans = mixture.BayesianGaussianMixture(
        n_components=10,
        covariance_type="full",
        weight_concentration_prior=1e2,
        weight_concentration_prior_type="dirichlet_process",
        mean_precision_prior=1e-2,
        covariance_prior=1e0 * np.eye(24, 24),
        init_params='kmeans',
        max_iter=200,
        n_init=5,
        random_state=2,
        verbose=1
    )
    #kmeans = GaussianMixture(n_components=optimal_k, random_state=42)
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
        'Endothelial': 3,'Immune': 3, 'Mesenchymal': 3,
        'Epithelial': 3
    }

    for cell_type in celltypes_selected:
        TAG = 'EXP_8'
        radii = '2'
        if cell_type in ['Granulocyte/mast']:
            cell_type = 'Granulocyte_mast'
        INPUT_FILE = f'{PATH}/band_features_{cell_type}_cell_id_br{radii}.csv'
        OUT_PATH = f'{OUTPATH}/br{radii}_{TAG}/{cell_type}'
        #OUT_PATH = f'{OUTPATH}/br2_custom/{cell_type}'
        mkdirs(OUT_PATH)

        #Moving the script to the folder to be analysed later for picking best parameters
        copy('/well/rittscher/users/qwi813/CellMiroEnviron/application/mine/band_features_clusters_tweaking.py',f'{OUTPATH}/br{radii}_{TAG}/band_features_clusters.py')
        PLOTS_DIR = f'{OUT_PATH}/plots'
        CSV_DIR = f'{OUT_PATH}/csv'
        mkdirs(PLOTS_DIR)
        mkdirs(CSV_DIR)
        print(f'Processing cell type {cell_type}')
        data = pd.read_csv(INPUT_FILE)
        print(cell_type, data.shape)
        # removing cells that are ignored due to density
        data = data[data.iloc[:,:-2].sum(1)>0]
        # idx = np.random.randint(0,len(data),500)
        # data = data.iloc[idx,:]
        features_col = list(set(data.columns) - {'sample_id', 'label','cell_id'})
        perform_clustering_and_generate_plots(data, features_col, cell_type,celltypes_selected = celltypes_selected)
