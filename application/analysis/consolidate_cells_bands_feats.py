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
K = 3
FIG_SIZE = (8, 10)

PATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull'
OUTPATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_Clusters'


# Main script execution
if __name__ == "__main__":

    TAG = 'EXP_5'
    radii = '1'

    CONSOL_FILE = f'{DATA_DIR}/{EXP_TAG}.csv'
    consolDf = pd.read_csv(CONSOL_FILE, index_col='Unnamed: 0')
    OUT_STATS_FILE = f'{OUTPATH}/br{radii}_{TAG}/dataset_with_envrion_feats.csv'

    celltypes_selected = {
        'Endothelial': 3,'Immune': 3, 'Mesenchymal': 3,
        'Epithelial': 3
    }

    for cell_type in celltypes_selected:

        print(f'Processing {cell_type}')
       
        if cell_type in ['Granulocyte/mast']:
            cell_type = 'Granulocyte_mast'
        CLUST_INPUT_FILE = f'{OUTPATH}/br{radii}_{TAG}/{cell_type}/csv/{cell_type}.csv'
        print(CLUST_INPUT_FILE)
        clustDf = pd.read_csv(CLUST_INPUT_FILE).drop(columns='Unnamed: 0')
        clustDf.index = clustDf['sample_id']+'_'+clustDf['cell_id']
        clustDf.rename(columns={'cell_id':'c_cell_id'}, inplace=True)
        clustDf['cluster'] = clustDf['cluster'].astype(str)  # Ensure it's all strings
        clustDf['ENVIRON'] = cell_type + '-' + clustDf['cluster']
        selected_indices = clustDf.index.tolist()
        selected_cols = clustDf.columns.tolist()
        consolDf.loc[selected_indices,selected_cols] = clustDf.loc[selected_indices,selected_cols].to_numpy()
    total_cells = consolDf.final_lineage.value_counts(dropna=False).sum()
    dropped = consolDf.ENVIRON.isna().sum()
    print(f'%age of cells dropped {(dropped/total_cells)*100}')
    consolDf.to_csv(OUT_STATS_FILE)

        
    

