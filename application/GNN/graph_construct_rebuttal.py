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
 compute_surface_distances
 )

if __name__ == '__main__':
    
    ROOT_DIR = PROJECT_DIR
    OUTPUT_DIR = OUTPUT_DIR   
    FEAT = 'CELL_TYPES'
    ENVIRON_COL = 'ENVIRON'
    CELL_COL = 'final_lineage'
    K = 5
    R_max = 600
    TAG = f'K_{K}_R_max_{R_max}'
    interaction_radius_microns = 50
    GRAPHS_DIR = f'{DATA_DIR}/GRAPHS/{TAG}_{interaction_radius_microns}'
    mkdirs(GRAPHS_DIR)
    OUTPATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_rebuttal'
    STATS_FILE = f'{OUTPATH}/dataset_band_desc_{TAG}.csv'
    df = pd.read_csv(STATS_FILE, index_col='Unnamed: 0')
    cell_types = df[CELL_COL].unique()
    band_feats = [col for col in df.columns if col.split('_')[0] in cell_types]

    #Dropping Cells with no ENVIRON FEATURES
    #df = df.dropna(subset=[ENVIRON_COL, CELL_COL])

    #Dropping with with empty band descriptor
    df = df.dropna(subset=band_feats+[CELL_COL])
    cellFeatDf = pd.get_dummies(df[CELL_COL], prefix = CELL_COL)*1
    
    #'environFeatDf = pd.get_dummies(df[ENVIRON_COL], prefix = ENVIRON_COL)*1

    cell_feats_cols = cellFeatDf.columns.tolist()
    #environ_feats_cols = environFeatDf.columns.tolist()
    df = pd.concat([df,cellFeatDf], axis=1)
    #df = pd.concat([df,cellFeatDf,environFeatDf], axis=1)
    #features_list = cell_feats_cols + environ_feats_cols + band_feats

    features_list = cell_feats_cols + band_feats
    
    for case in set(df['sample']):
        out_graph_file = f'{GRAPHS_DIR}/{case}.json'
        sampDf = df[df['sample']==case]
        generate_cell_adjacency_graph(sampDf,out_graph_file,
                    interaction_radius_microns = interaction_radius_microns,
                    features=features_list)# features)#CELL_TYPES_NAMES++['distance_to_bone_binary']
 