from ast import arg
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.spatial import distance_matrix, Delaunay
import random
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from model import *
import argparse
from pathlib import Path
from helper_functions import load_graph_from_json
import sys
sys.path.append('.')

from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir
from application.utils.shape_approx import calculate_points_density, alpha_shape

from helper_functions import (
load_counts_matrix,
 select_within_polygons,
 create_geojson,
 load_and_preprocess_annot,
 combine_geojson,
 save_geojson,
 generate_cell_adjacency_graph,
 read_it_regions_annotations,
 mkdirs,
 read_geojson,
 transform_coordinates,
 transform_geometries,
 compute_surface_distances,
 fast_visualize_graph
 )

if __name__ == '__main__':
    
    ROOT_DIR = PROJECT_DIR
    OUTPUT_DIR = OUTPUT_DIR   
    FEAT = 'CELL_TYPES'
    ENVIRON_COL = 'ENVIRON'
    CELL_COL = 'final_lineage'
    TAG = 'EXP_5'
    radii = '1'
    interaction_radius_microns = 50
    GRAPHS_DIR = f'{DATA_DIR}/GRAPHS/{TAG}_{radii}_{interaction_radius_microns}'
    GRAPHS_DIR_VIZ = f'{DATA_DIR}/GRAPHS_VIZ/{TAG}_{radii}_{interaction_radius_microns}'
    mkdirs(GRAPHS_DIR_VIZ)

    graphlist = glob(os.path.join(GRAPHS_DIR, "*.json"))#[0:1000]
    graphDf = pd.DataFrame(graphlist,columns=['Path'])
    graphDf['sample_key'] = [Path(g).stem for g in graphDf['Path']]
    graphDf.set_index('sample_key',inplace=True)

    for sample in tqdm(graphDf.index):
        print(f'Processing sample {sample}')
        file_path = graphDf.loc[sample,'Path']
        graph_dict = load_graph_from_json(file_path)
        out_file = f'{GRAPHS_DIR_VIZ}/{sample}.png'
        fast_visualize_graph(graph_dict=graph_dict, outPath=out_file)
        