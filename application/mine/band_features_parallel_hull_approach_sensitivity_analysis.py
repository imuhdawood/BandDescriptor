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

from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir
from application.utils.shape_approx import calculate_points_density, alpha_shape

def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def make_band_edges(r_max: float, k: int) -> np.ndarray:
    """
    Equal-width concentric-band edges for a given outer radius and band count.

    Parameters
    ----------
    r_max : float
        Outer radius (µm, px, or any consistent distance unit).
    k : int
        Number of bands.

    Returns
    -------
    np.ndarray
        Array of length k+1 with edges `[0, Δr, 2Δr, …, r_max]`,
        where Δr = r_max / k.
    """
    return np.linspace(0, r_max, k + 1)


def compute_microenvironment(cell_idx, radii, data, tree, cell_bands_template, mapping):
    center = data[['x', 'y']].loc[cell_idx].values.reshape(1, -1)
     # Initialize a dictionary to hold counts
    cell_bands = cell_bands_template.copy()
    
    # Relaxing this constraint as now we want to analyse very narrow and large band
    # for the performance of predictive model


    # # Excluding cells in very low density areas
    # idx_within_max = tree.query_radius(center, r=np.median(radii))[0]
    # # print(f'max cell counts {len(idx_within_max)}')
    # if len(idx_within_max) < 50:
    #     return cell_bands  # Return 0 values for all band feats
    
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

def process_sample(sample_id, consolDf, cell_of_interest, celltypes_selected, radii, radius=150, density_threshold=20, alpha=10, inner_strip_width=100):

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

    CONSOL_FILE = f'{DATA_DIR}/{EXP_TAG}.csv'
    consolDf = pd.read_csv(CONSOL_FILE, index_col='Unnamed: 0')

    selected_columns = ['sample_key', 'cell_type','x','y','cell_id']
    mapping = dict( 
        zip(
                ['sample', ANNOT_LEVEL, 'x_centroid', 'y_centroid', 'cell_id' ],
                selected_columns
            )
    )
    consolDf.rename(
       columns=mapping, inplace=True
    )

    cases = list(set(consolDf['sample_key']))

    #cases = cases[0:2] # Test run

    OUT_DIR = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_rebuttal'
    mkdir(OUT_DIR)
    # 'Mesenchymal', 'Epithelial'
    run = [
       'Mesenchymal', 'Epithelial','Immune', 'Endothelial'
    ]

    celltypes_selected = [
        'Immune', 'Endothelial',
        'Mesenchymal', 'Epithelial'
    ]

    #I am selecting a value of K which is the number of bands and R_max which is the 
    # value of maximal covering radius
    outer_radii = [300, 500, 600]   # µm
    band_counts = [1, 3, 5]

    # idx = 2

    # Running 
    # K=1, R_max = 300
    #K = 3, R_max = 300
    #K = 5, R_max = 300

    #K=1, R_max = 500
    #K=3, R_max = 500
    #K=5, R_max = 500
    
    #K=1, R_max = 600
    #K=5, R_max = 600

    K = 3
    R_max = 600
    # K = band_counts[idx]
    # R_max = outer_radii[idx]



    radii = make_band_edges(R_max, K)

    radii = [round(r) for r in radii[1:]]

    print(f'Value of K {K} and R_max {R_max}')
    print(f'Selected radial bands', radii)

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
        if cell_of_interest in ['SMCs/Pericytes']:
            cell_of_interest = 'SMCs_Pericytes'

        output_file = f'{OUT_DIR}/band_features_{cell_of_interest}_cell_id_K_{K}_Rmax_{R_max}.csv'
        allSampleDf.to_csv(output_file, index=False)
        print(f'Saved results to {output_file}')

if __name__ == "__main__":
    main()
