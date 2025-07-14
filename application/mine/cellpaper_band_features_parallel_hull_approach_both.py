import json
import os
import multiprocessing as mp
from typing import List
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
import sys
sys.path.append('.')

from application.project_cp import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir
from application.utils.shape_approx import calculate_points_density, alpha_shape

def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def compute_microenvironment(cell_idx, radii, data, tree, cell_bands_template, mapping):
    center = data[['x', 'y']].loc[cell_idx].values.reshape(1, -1)
     # Initialize a dictionary to hold counts
    cell_bands = cell_bands_template.copy()
    
    # Excluding cells in very low density areas
    idx_within_max = tree.query_radius(center, r=500)[0]
    # print(f'max cell counts {len(idx_within_max)}')
    if len(idx_within_max) < 50:
        return cell_bands  # Return 0 values for all band feats
    
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

def process_sample(sample_id, consolDf, cell_of_interest, celltypes_selected, radii, radius=500, density_threshold=50, alpha=10, inner_strip_width=100):

    fDf = consolDf[consolDf.sample_key.isin([sample_id])]
    data = fDf[['x', 'y','cell_type','cell_id']]
    data = data[data.cell_type.isin(celltypes_selected)]
    cell_bands_template = {f'{c}_{r}': 0 for c in celltypes_selected for r in radii}
    # Build KD-Tree for spatial queries
    centroids = data[['x', 'y']].values
    cell_ids = data.index.tolist()
    tree = KDTree(centroids)

    # Computing point spatial density and select cells for which to define the band descriptor
    densities = tree.query_radius(centroids, r=radius, count_only=True)
    filtered_centroids = centroids[densities >= density_threshold]
    print(f'{len(filtered_centroids)} Cells selected based on points density')

    # Concave Hull around filtered points
    concave_hull = alpha_shape(filtered_centroids, alpha)
    cells_in_hull = [barcode for idx,barcode in enumerate(cell_ids) if concave_hull.contains(Point(centroids[idx]))]
    print(f'{len(cells_in_hull)} Cells are inside Concave Hull')

    # Create an inner boundary by shrinking the concave hull
    inner_boundary = concave_hull.buffer(-inner_strip_width)
    safe_idx = [barcode for idx,barcode in enumerate(cell_ids) if inner_boundary.contains(Point(centroids[idx]))]
    print(f'{len(safe_idx)} Cells after inner safe strip')

    mapping = dict(
        zip(
            list(range(data.shape[0])),
            data.index.tolist()
    ))
    # Filter only the cells of interest within the safe boundary
    no_filtering = data['cell_type'] == cell_of_interest
    print('Without boundary check',sum(no_filtering))

    filter_data = data.loc[safe_idx,:]
    subset_idx = filter_data[filter_data['cell_type']==cell_of_interest].index.to_numpy()
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

    CONSOL_FILE = f'{DATA_DIR}/cell_paper_data.csv'
    consolDf = pd.read_csv(CONSOL_FILE)

    selected_columns = ['sample_key', 'cell_type','x','y','cell_id']
    mapping = dict(
        zip(
            ['H_field', ANNOT_LEVEL, 'x.coord', 'y.coord', 'Unnamed: 0'],
            selected_columns
        )
    )
    consolDf.rename(
       columns=mapping, inplace=True
    )
    consolDf = consolDf[~consolDf['cell_type'].isin(['Artifact'])]
    cases = list(set(consolDf['sample_key']))

    OUT_DIR = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull'
    mkdir(OUT_DIR)
    # 'Mesenchymal', 'Epithelial'
    run = [
    #     'Myeloid', 'Lymphoid', 'Monocytes', 'Adipocyte',
    #    'Endosteal', 'SEC', 'Erythroid', 'Megakaryocyte',
        'HSPC', 'MSC','Macrophages', 'AEC',
        'pDC', 'VSMC', 'Schwann Cells'
       ]

    celltypes_selected = [
        'Myeloid', 'Lymphoid', 'Monocytes', 'Adipocyte',
       'Endosteal', 'SEC', 'Erythroid', 'Megakaryocyte', 'HSPC', 'MSC',
       'Macrophages', 'AEC', 'pDC', 'VSMC', 'Schwann Cells'
    ]

    #celltypes_selected = consolDf['cell_type'].tolist()


    #radii = [100, 150, 200, 250, 300, 350, 400, 450, 500]# These are band radii 1 clusters
    #radii = [50,100, 200, 300, 400, 500]# These are band radii 1 clusters

    radii = [100, 200, 300, 400, 500] # Second Raddi Band R1

    for cell_of_interest in run:
        print('Processing *******************************************')
        print('********************************************************')
        print(cell_of_interest)
        allSampleDf = pd.DataFrame()

        for sample_id in tqdm(cases):
            mDf = consolDf[consolDf.sample_key.isin([sample_id])].loc[:,selected_columns]
            print(mDf.dropna().shape, sample_id)

            features_df = process_sample(
                sample_id,consolDf,
                cell_of_interest, celltypes_selected,
                radii, alpha=500
                )
            allSampleDf = pd.concat([allSampleDf, features_df], ignore_index=True)
        if cell_of_interest in ['SMCs/Pericytes']:
            cell_of_interest = 'SMCs_Pericytes'

        output_file = f'{OUT_DIR}/band_features_{cell_of_interest}_cell_id_radii_{str(radii[0])}.csv'
        allSampleDf.to_csv(output_file, index=False)
        print(f'Saved results to {output_file}')

if __name__ == "__main__":
    main()
