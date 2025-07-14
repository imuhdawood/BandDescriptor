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
OUTPATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_rebuttal'


# Main script execution
if __name__ == "__main__":

    K = 5
    R_max = 600
    TAG = f'K_{K}_R_max_{R_max}'

    CONSOL_FILE = f'{DATA_DIR}/{EXP_TAG}.csv'
    consolDf = pd.read_csv(CONSOL_FILE, index_col='Unnamed: 0')
    OUT_STATS_FILE = f'{OUTPATH}/dataset_band_desc_{TAG}.csv'

    celltypes_selected = [
        'Endothelial','Immune', 'Mesenchymal', 'Epithelial'
    ]

    for cell_type in celltypes_selected:

        print(f'Processing {cell_type}')
       
        if cell_type in ['Granulocyte/mast']:
            cell_type = 'Granulocyte_mast'
        BAND_DESC_INPUT_FILE = f'{OUTPATH}/band_features_{cell_type}_cell_id_K_{K}_Rmax_{R_max}.csv'
        print(BAND_DESC_INPUT_FILE)
        bandDescDf = pd.read_csv(BAND_DESC_INPUT_FILE)
        bandDescDf.index = bandDescDf['sample_id'] + '_' + bandDescDf['cell_id']
        bandDescDf.rename(columns={'cell_id':'c_cell_id'}, inplace=True)
        selected_indices = bandDescDf.index.tolist()
        selected_cols = bandDescDf.columns.tolist()
        consolDf.loc[selected_indices,selected_cols] = bandDescDf.loc[selected_indices,selected_cols].to_numpy()
    #total_cells = consolDf.final_lineage.value_counts(dropna=False).sum()
    consolDf.to_csv(OUT_STATS_FILE)

        
    

