import json
import os
import multiprocessing as mp
from typing import List
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import sys
sys.path.append('.')

from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL


def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def compute_microenvironment(cell_idx, radii, data, tree, cell_bands_template, mapping):
    center = data[['x', 'y']].loc[cell_idx].values.reshape(1, -1)
    
    # Initialize a dictionary to hold counts
    cell_bands = cell_bands_template.copy()

    previous_radius = 0
    for r in radii:
        # Query cells within the current radius
        idx_within_radius = tree.query_radius(center, r=r)[0]

        # Get only the cells in the current band, not overlapping with previous radius
        if previous_radius > 0:
            idx_within_previous = tree.query_radius(center, r=previous_radius)[0]
            idx_in_band = np.setdiff1d(idx_within_radius, idx_within_previous)
        else:
            idx_in_band = idx_within_radius

        if len(idx_in_band) > 0:
            # Count the occurrences of each cell type within the band
            idx_in_band = [mapping[id] for id in idx_in_band]
            type_counts = data.loc[idx_in_band]['cell_type'].value_counts(normalize=True)
            for cell_type, count in type_counts.items():
                cell_bands[f'{cell_type}_{r}'] = count
        previous_radius = r
    cell_bands['cell_id'] = data.loc[cell_idx].cell_id    
    return cell_bands

def process_sample(sample_id, consolDf, cell_of_interest, celltypes_selected, radii):

    fDf = consolDf[consolDf.sample_key.isin([sample_id])]
    data = fDf[['x', 'y','cell_type','cell_id']]
    data = data[data.cell_type.isin(celltypes_selected)]
    cell_bands_template = {f'{c}_{r}': 0 for c in celltypes_selected for r in radii}
    # Build KD-Tree for spatial queries
    tree = KDTree(data[['x', 'y']].values)
    # Define the safe boundary based on the maximum radius
    max_radius = 100#max(radii) # These are TMA cores we need to be a bit careful max is too much
    x_min_safe, y_min_safe = max_radius, max_radius
    x_max_safe, y_max_safe = data['x'].max() - max_radius, data['y'].max() - max_radius
    mapping = dict(
        zip(
            list(range(data.shape[0])),
            data.index.tolist()
    ))
    # Filter only the cells of interest within the safe boundary
    no_filtering = data['cell_type'] == cell_of_interest
    print('Without boundary check',sum(no_filtering))
    subset_idx = data[
        (data['cell_type'] == cell_of_interest) &
        (data['x'] > x_min_safe) & (data['x'] < x_max_safe) &
        (data['y'] > y_min_safe) & (data['y'] < y_max_safe)
    ].index.to_numpy()
    print('After filtering', len(subset_idx))
    
    # Parallelize the microenvironment computation
    with mp.Pool(mp.cpu_count()) as pool:#
        features = pool.starmap(
            compute_microenvironment, 
            [(idx, radii, data, tree, cell_bands_template, mapping) for idx in subset_idx]
        )
    
    features_df = pd.DataFrame(features)
    features_df['sample_id'] = sample_id

    return features_df

def main():

    CONSOL_FILE = f'{DATA_DIR}/{EXP_TAG}.csv'
    consolDf = pd.read_csv(CONSOL_FILE)

    selected_columns = ['sample_key', 'cell_type','x','y','cell_id']
    mapping = dict( 
        zip(
                ['sample', ANNOT_LEVEL, 'x_centroid', 'y_centroid', 'Unnamed: 0' ],
                selected_columns
            )
    )
    consolDf.rename(
       columns=mapping, inplace=True
    )

    cases = list(set(consolDf['sample_key']))

    OUT_DIR = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_S100'
    mkdir(OUT_DIR)

    run = [
        'Mesenchymal', 'Epithelial'
        ]

    #run = ['MNP', 'Lymphocyte', 'Stromal']
    celltypes_selected = [
        'Immune', 'Endothelial',
        'Mesenchymal', 'Epithelial'
    ]


    #radii = [100, 150, 200, 250, 300, 350, 400, 450, 500]# These are band radii 1 clusters
    radii = [100, 200, 300, 400, 500]# These are band radii 1 clusters

    for cell_of_interest in run:
        print('Processing *******************************************')
        print('********************************************************')
        print(cell_of_interest)
        allSampleDf = pd.DataFrame()

        for sample_id in tqdm(cases):
            mDf = consolDf[consolDf.sample_key.isin([sample_id])].loc[:,selected_columns]
            print(mDf.dropna().shape, sample_id)

            features_df = process_sample(sample_id,consolDf, cell_of_interest, celltypes_selected, radii)
            allSampleDf = pd.concat([allSampleDf, features_df], ignore_index=True)
        if cell_of_interest in ['Granulocyte/mast']:
            cell_of_interest = 'Granulocyte_mast'

        output_file = f'{OUT_DIR}/band_features_{cell_of_interest}_cell_id_br2.csv'
        allSampleDf.to_csv(output_file, index=False)
        print(f'Saved results to {output_file}')

if __name__ == "__main__":
    main()
