import numpy as np
import cv2
import scipy.io as sio
import gzip
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import median_abs_deviation
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
from scipy.sparse import lil_matrix
import geopandas as gpd
import json
from shapely.geometry import Point, Polygon
from sklearn.neighbors import radius_neighbors_graph
from typing import Optional, Union, List,Dict,Any
from torch_geometric.data import Data
import random
from tqdm import tqdm

import pyvips
import tifftools

import glob,os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def update_pidx(dataset):
    """
    Updates the 'spidx' attribute for each graph in the dataset.

    Parameters:
    dataset (list): List of graph objects to be updated.
    """
    for i in range(len(dataset)):
        G = dataset[i]
        G.spidx = f'{G.pid}-{G.sid}'
        dataset[i] = G

def update_node_predictions(graphs, node_pred_dict, node_predictions, end_point, device='cpu'):
    """
    Update the node_predictions dictionary with nodeproba and other information based on the given graphs.

    Args:
    - graphs (iterable): A list of graph objects, where each graph contains information such as edge_index, y, and spidx.
    - node_pred_dict (dict): A dictionary where the keys are sample identifiers (spidx) and the values are lists or arrays containing the predicted probabilities or values to be added to the 'nodeproba' field.
    - node_predictions (dict): A dictionary that holds graphs indexed by their 'spidx' (or other unique identifier). Each graph can have attributes such as 'nodeproba'.
    - end_point (str): The class label to assign to each graph's 'class_label' attribute.
    - device (str): The device where the graph should be moved for processing ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
    - node_predictions (dict): The updated dictionary of graphs, with each graph having its 'nodeproba' field updated (or created) and 'class_label' set to the given 'end_point'.
    """
    for i, GG in enumerate(tqdm(graphs)):
        sample_id = GG.spidx 
        zxn = node_pred_dict[sample_id][0]  # Extract node predictions

        # If the sample_id is not already in node_predictions, create a new entry
        if sample_id not in node_predictions:
            G = Data(edge_index=GG.edge_index, y=GG.y, spidx=GG.spidx)  # Create a new graph
            G.to(device) 
            G.nodeproba = zxn  
            G.class_label = end_point
            node_predictions[sample_id] = G  
        else:
            # If the sample_id already exists, append the new predicted value to nodeproba
            G = node_predictions[sample_id]
            G.nodeproba = np.dstack((G.nodeproba, zxn)) 
            node_predictions[sample_id] = G

    return node_predictions


def generate_statistical_reports(all_predictions_df, consolDf, results_dir, stats=['mean', 'median', 'sum']):
    """
    Generate statistical reports (mean, median, sum) of the predicted scores grouped by Patient ID.

    Args:
    - all_predictions_df (pd.DataFrame): DataFrame containing predictions with columns 'Patient ID' and 'Predicted Score'.
    - consolDf (pd.DataFrame): Another DataFrame to join with 'all_predictions_df', likely containing additional information.
    - results_dir (str): Directory where the CSV files should be saved.
    - stats (list): List of statistics to compute (e.g., ['mean', 'median', 'sum']). Default is ['mean', 'median', 'sum'].

    Returns:
    - None: Saves the resulting reports as CSV files in the specified directory.
    """
    # Group by 'Patient ID' and calculate the required statistics
    for stat in stats:
        if stat == 'mean':
            statDf = all_predictions_df.groupby('Patient ID')['Predicted Score'].mean().reset_index()
        elif stat == 'median':
            statDf = all_predictions_df.groupby('Patient ID')['Predicted Score'].median().reset_index()
        elif stat == 'sum':
            statDf = all_predictions_df.groupby('Patient ID')['Predicted Score'].sum().reset_index()
        else:
            raise ValueError(f"Unsupported statistic: {stat}")

        # Adjust index based on 'Patient ID'
        statDf.index = [k.split('-')[-1] for k in statDf['Patient ID']]

        # Join with consolDf and sort by the computed statistic
        statDf_sorted = statDf.join(consolDf).sort_values(by='Predicted Score', ascending=True)

        # Save the result to a CSV file
        stat_file = f'{results_dir}/{stat}_prediction_scores_all.csv'
        statDf_sorted.to_csv(stat_file, index=False)
        print(f"Saved {stat} prediction scores to {stat_file}")


def fast_visualize_graph(graph_dict, sample_size=500000, outPath=None):
    """
    Efficiently visualizes large graphs by optimizing NetworkX drawing.
    
    Parameters:
    - graph_dict (dict): Dictionary containing 'pos' and 'edge_index'.
    - sample_size (int): Number of nodes to sample for visualization.
    """

    # Extract node positions and edges
    coordinates = np.array(graph_dict['pos'])
    edge_index = np.array(graph_dict['edge_index'])

    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes (sample if too large)
    num_nodes = len(coordinates)
    if num_nodes > sample_size:
        sampled_indices = np.random.choice(num_nodes, sample_size, replace=False)
    else:
        sampled_indices = np.arange(num_nodes)

    for i in sampled_indices:
        G.add_node(i, pos=(coordinates[i, 0], coordinates[i, 1]))

    # Add edges (only between sampled nodes)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        if src in sampled_indices and dst in sampled_indices:
            G.add_edge(src, dst)

    # Use precomputed positions
    pos = {i: (coordinates[i, 0], coordinates[i, 1]) for i in sampled_indices}

    # Faster drawing using scatter + draw_networkx_edges
    plt.figure(figsize=(10, 8))
    
    # Draw edges separately
    nx.draw_networkx_edges(G, pos, alpha=0.7, width=0.6)

    # Use scatter for nodes (much faster than nx.draw)
    coords = np.array(list(pos.values()))
    plt.scatter(coords[:, 0], coords[:, 1], s=1, color="blue", alpha=0.7)

    plt.title(f"Cell Adjacency Graph ({len(sampled_indices)} nodes)")
    plt.savefig(outPath)


def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

def quality_control(adata):
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))
    
    sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )
    
    adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )
    
    # filtering mitocondial genes with percentage above 8
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
    adata.obs["pct_counts_mt"] > 8
    )
    
    print(f"Total number of cells: {adata.n_obs}")
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

    print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")
    
    return adata
    
def select_within_polygons(centroids_df, polygons_df, key='Selection'):
    """
    Selects points from centroids_df that fall within any polygons defined in polygons_df.
    
    Parameters:
    - centroids_df (pd.DataFrame): DataFrame with columns 'x_centroid' and 'y_centroid'.
    - polygons_df (pd.DataFrame): DataFrame containing polygon coordinates and selection keys.
    - key (str): Column name in polygons_df representing unique selections (default: 'Selection').
    
    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing only centroids that fall within any polygons.
    """
    # Create a GeoDataFrame from centroids
    gdf_centroids = gpd.GeoDataFrame(
        centroids_df, 
        geometry=[Point(xy) for xy in zip(centroids_df.x_centroid, centroids_df.y_centroid)],
        crs="EPSG:4326"  # Set a coordinate reference system; adjust as necessary
    )
    
    # Create a list of polygon geometries
    polygon_geometries = []
    for selection in polygons_df[key].unique():
        subset = polygons_df[polygons_df[key] == selection]
        coordinates = subset[['X', 'Y']].values.tolist()
        polygon = Polygon(coordinates)
        polygon_geometries.append(polygon)

    # Create a GeoDataFrame for polygons
    gdf_polygons = gpd.GeoDataFrame({
        key: polygons_df[key].unique(),
        'geometry': polygon_geometries
    }, crs="EPSG:4326")  # Ensure the CRS matches the centroids CRS
    
    # Perform spatial join: find centroids within polygons
    joined = gpd.sjoin(gdf_centroids, gdf_polygons, how="inner", predicate='within')

    return joined 


def get_morph_features(cnt):
    """Get morphological features contour
       For details refer to:
       https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    """


    # Calculating Image monets from Contours:
    M = cv2.moments(cnt)

    # ** regionprops **
    area = M["m00"]
    if area > 0:
        
        # Bouding box return: x,y,w,h 
        bBox = cv2.boundingRect(cnt)

        # The shape is close contour so passing True
        # Return perimiter
        perimeter = cv2.arcLength(cnt, True)
      
        # centroid    = m10/m00, m01/m00 (x,y)
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])

        # EquivDiameter: diameter of circle with same area as region
        equivDiameter = np.sqrt(4 * area / np.pi)

        # Extent: ratio of area of region to area of bounding box
        extent = area / (bBox[2] * bBox[3])

        # convex hull vertices
        hull = cv2.convexHull(cnt)
        convexArea = cv2.contourArea(hull)

        # Solidity 
        solidity = area / convexArea

        # ELLIPSE - determine best-fitting ellipse.
        centre, axes, angle = cv2.fitEllipse(cnt)
        maj = np.argmax(axes)  # this is MAJor axis, 1 or 0
        min = 1 - maj  # 0 or 1, minor axis
        # Note: axes length is 2*radius in that dimension
        majorAxisLength = axes[maj]
        minorAxisLength = axes[min]
        eccentricity = np.sqrt(1 - (axes[min] / axes[maj]) ** 2)
        orientation = angle
        ellipseCentre = centre  # x,y

    else:
        perimeter = 0
        extent = 0
        equivDiameter = 0
        solidity = 0
        convexArea = 0
        majorAxisLength = 0
        minorAxisLength = 0
        eccentricity = 0
        orientation = 0
        ellipseCentre = [0,0]

    return {
        "area": area,
        "perimeter": perimeter,
        'extent':extent,
        'equivalent-diameter':equivDiameter,
        'solidity':solidity,
        'convex-area':convexArea,
        'major-axis-length':majorAxisLength,
        'minor-axis-length':minorAxisLength,
        'eccentricity':eccentricity,
        'orientation':orientation,
        'ellipse-centre-x':ellipseCentre[0],
        'ellipse-centre-y':ellipseCentre[1]
    }


import json
from typing import Dict, Optional

def save_geojson(data: Dict, filename: str, indent: Optional[int] = 2) -> None:
    """
    Saves a GeoJSON object to a specified file with optional indentation for pretty printing.

    Parameters:
    - data (Dict): The GeoJSON data to be saved, must be a dictionary.
    - filename (str): The path to the file where the GeoJSON data should be saved.
    - indent (int, optional): The indentation level for formatting the output file. Default is 2.

    Returns:
    - None
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)
    print(f"GeoJSON data successfully saved to {filename}")

def read_geojson(file_path):
    """
    Reads a GeoJSON file and returns a GeoDataFrame.

    Parameters:
    file_path (str): The path to the GeoJSON file.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the geometries and attributes from the GeoJSON file.
    """
    try:
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        print(f"Error reading GeoJSON file {file_path}: {e}")
        return None

def combine_geojson(geojson1, geojson2):
    combined_geojson = {
        "type": "FeatureCollection",
        "features": geojson1["features"] + geojson2["features"]
    }
    return combined_geojson


def transform_coordinates(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply an affine transformation to a set of 2D coordinates.

    Parameters:
    coords (np.ndarray): A numpy array of shape (n, 2), where n is the number of coordinates.
                         Each row represents a 2D coordinate (x, y).
    matrix (np.ndarray): A 3x3 affine transformation matrix.

    Returns:
    np.ndarray: A numpy array of shape (n, 2) containing the transformed coordinates.
    """
    # Convert to homogeneous coordinates by adding a column of ones
    ones = np.ones((coords.shape[0], 1))
    homogeneous_coords = np.hstack([coords, ones])
    
    # Apply the affine transformation matrix
    transformed_coords = homogeneous_coords @ matrix.T
    
    # Convert back to 2D coordinates
    return transformed_coords[:, :2]

def transform_geometries(gdf: gpd.GeoDataFrame, matrix: np.ndarray) -> gpd.GeoDataFrame:
    """
    Apply an affine transformation to the geometries in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing the geometries to transform.
    matrix (np.ndarray): A 3x3 affine transformation matrix.

    Returns:
    GeoDataFrame: A new GeoDataFrame with the transformed geometries.
    """
    transformed_geometries = []

    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            # Extract and transform exterior coordinates
            exterior_coords = np.array(geom.exterior.coords)
            transformed_exterior_coords = transform_coordinates(exterior_coords, matrix)
            
            # Transform interior coordinates (holes) if any
            transformed_interiors = [transform_coordinates(np.array(interior.coords), matrix) for interior in geom.interiors]
                
            # Create new transformed polygon
            transformed_polygon = Polygon(transformed_exterior_coords, transformed_interiors)
            transformed_geometries.append(transformed_polygon)
        else:
            # Handle other geometry types if necessary
            transformed_geometries.append(geom)
    
    # Create a transformed GeoDataFrame
    gdf_transformed = gdf.copy()
    gdf_transformed['geometry'] = transformed_geometries
    
    return gdf_transformed

def compute_surface_distances(cells_gdf: gpd.GeoDataFrame, bones_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Compute the minimum distance from each cell centroid to the nearest bone surface.

    Parameters:
    cells_gdf (gpd.GeoDataFrame): A GeoDataFrame containing cell centroids as points.
    bones_gdf (gpd.GeoDataFrame): A GeoDataFrame containing bone geometries.

    Returns:1;'
    np.ndarray: An array of minimum distances from each cell centroid to the nearest bone surface.
    """
    #cells_gdf = cells_gdf.to_crs(bones_gdf.crs)
    projected_crs = "EPSG:32633"
    return cells_gdf.geometry.apply(lambda cell: bones_gdf.distance(cell).min())

def load_and_preprocess_annot(input_csv_path, resolution=None):
    """
    Load a CSV file, skip the first two rows, and adjust coordinates based on the provided resolution.
    
    Parameters:
    - input_csv_path (str): Path to the input CSV file.
    - resolution (float, optional): The factor by which to divide the X and Y coordinates. 
                                    Defaults to None, which will skip scaling coordinates.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame with adjusted coordinates if resolution is provided.
    """
    df = pd.read_csv(input_csv_path, skiprows=2)
    if resolution is not None:
        df[['X', 'Y']] = df[['X', 'Y']] / resolution
    return df


def load_counts_matrix(cell_expr_dir=None):
    
    """Input: 
        Accept directory of xenium experiment containing cell level transcript information

    Returns:
        Return a DataFrame of cell level transcripts
    """
    # Open the compressed file
    with gzip.open(f'{cell_expr_dir}/matrix.mtx.gz', 'rb') as f:
        # Load the sparse matrix
        sparse_matrix = sio.mmread(f)
    # Convert the sparse matrix to a dense array
    dense_array = sparse_matrix.toarray()
    # Reading List of genes
    genes = pd.read_csv(f'{cell_expr_dir}/features.tsv.gz',
                        delimiter='\t',
                        header=None
                        )
    selected = genes.iloc[:,-1]=='Gene Expression'
    dense_array = dense_array[selected,:]
    genes = genes[selected].iloc[:,1].tolist()
    barcodes = pd.read_csv(f'{cell_expr_dir}/barcodes.tsv.gz',
                           delimiter='\t',
                            header=None
                            ).iloc[:,0].tolist()
    
    cell_expr = pd.DataFrame(dense_array.T,columns=genes,
                 index=barcodes)
    return cell_expr


# GRAPH CONTRUCTION RELATED FUNCTION

def adjacency_delaunay(CC, distance=15):
    """
    Constructs a sparse adjacency matrix based on cluster connections within a distance threshold.

    Parameters:
    - CC (numpy.ndarray or list): Either Delaunay tessellation points or simple coordinate points.
    - dthresh (float): Distance threshold for considering connections between nodes.

    Returns:
    - W (scipy.sparse.lil_matrix): Sparse affinity matrix representing connections between clusters.
    """

    if isinstance(CC, np.ndarray):
        # If Cc_or_coordinates is a numpy array, assume it's Delaunay tessellation points
        tess = Delaunay(CC)
        clusters = CC
    else:
        # Otherwise, assume it's a list or numpy array of coordinates
        clusters = np.array(CC)
        tess = Delaunay(clusters)

    neighbors = defaultdict(set)
    
    # Build neighbor relationships based on Delaunay tessellation
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    
    # Create a sparse adjacency matrix
    W = lil_matrix((clusters.shape[0], clusters.shape[0]), dtype=np.float64)
    
    # Build adjacency matrix based on distance threshold
    for n in neighbors:
        neighbors[n] = np.array(list(neighbors[n]), dtype='int')
        neighbors[n] = neighbors[n][KDTree(clusters[neighbors[n], :]).query_ball_point(clusters[n], r=distance)]
        W[n, neighbors[n]] = 1.0
        W[neighbors[n], n] = 1.0
        
    return W

def filter_graph_data_by_random_region_removal(data, removal_fraction=0.2):
    """
    Randomly removes a subset of regions from the Data object and updates edge indices to preserve connectivity.

    Args:
        data (Data): The original Data object containing attributes like 'pos', 'edge_index', 'barcode', and 'IT_Region'.
        removal_fraction (float): The fraction of unique regions to remove. Defaults to 0.2.

    Returns:
        Data: The filtered Data object with a subset of regions removed and edge indices updated.
    """
    # print(data)
    # # Get unique regions
    # print(len(data.IT_Region))
    unique_regions = list(set(data.IT_Region))
    # print(unique_regions)
    
    # Determine number of regions to remove
    num_to_remove = int(len(unique_regions) * removal_fraction)
    
    # Randomly select regions to remove
    regions_to_remove = random.sample(unique_regions, num_to_remove)
    
    # Get the indices of the rows to keep
    mask = ~np.isin(data.IT_Region, regions_to_remove)
    
    # Create a new Data object as a copy of the original
    new_data = data.clone()

    # Filter nodes and attributes
    new_data.pos = data.pos[mask]
    new_data.barcode = np.array(data.barcode)[mask].tolist()
    new_data.IT_Region = np.array(data.IT_Region)[mask].tolist()
    
    if data.x is not None:
        new_data.x = data.x[mask]
    
    # Create a mapping from old indices to new indices
    index_mapping = torch.zeros(len(mask), dtype=torch.long)
    index_mapping[mask] = torch.arange(mask.sum())
    
    # Filter edge_index to keep only edges between retained nodes and reindex them
    edge_index = data.edge_index
    filtered_edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    new_data.edge_index = index_mapping[edge_index[:, filtered_edge_mask]]
    
    # Return the new Data object
    return new_data


def filter_graph_dict_by_random_region_removal(graph_dict, removal_fraction=0.2):
    """
    Randomly removes a subset of regions from the graph dictionary and updates edge indices to preserve connectivity.

    Args:
        graph_dict (dict): The original graph dictionary containing keys like 'coordinates', 'edge_index', 'barcode', and 'IT_Region'.
        removal_fraction (float): The fraction of unique regions to remove. Defaults to 0.2.

    Returns:
        dict: The filtered graph dictionary with a subset of regions removed and edge indices updated.
    """
    # Get unique regions
    unique_regions = list(set(graph_dict['IT_Region']))
    
    # Determine number of regions to remove
    num_to_remove = int(len(unique_regions) * removal_fraction)
    
    # Randomly select regions to remove
    regions_to_remove = random.sample(unique_regions, num_to_remove)
    
    # Get the indices of the rows to keep
    indices_to_keep = [i for i, region in enumerate(graph_dict['IT_Region']) if region not in regions_to_remove]
    
    # Create a mapping from old indices to new indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(indices_to_keep)}
    
    # Filter edge_index to keep only edges between retained nodes and reindex them
    filtered_edge_index = [
        [index_mapping[edge[0]], index_mapping[edge[1]]] for edge in np.array(graph_dict['edge_index']).T
        if edge[0] in index_mapping and edge[1] in index_mapping
    ]
    
    # Create the filtered graph dictionary
    filtered_graph_dict = {
        'coordinates': np.array(graph_dict['coordinates'])[indices_to_keep].tolist(),
        'edge_index': np.array(filtered_edge_index).T.tolist(),
        'barcode': np.array(graph_dict['barcode'])[indices_to_keep].tolist(),
        'IT_Region': np.array(graph_dict['IT_Region'])[indices_to_keep].tolist(),
    }
    
    if 'feats' in graph_dict:
        filtered_graph_dict['feats'] = np.array(graph_dict['feats'])[indices_to_keep,:].tolist()
        filtered_graph_dict['feat_names'] = graph_dict['feat_names']
    
    return filtered_graph_dict

def generate_cell_adjacency_graph(
    df: str, 
    output_file: str, 
    interaction_radius_microns: float = 15.0, 
    resolution_microns_per_pixel: float = 0.2125,
    linkage = 'Delaunay', 
    transcript_threshold: Optional[int] = None, 
    sample_size: Optional[int] = None,
    features:Optional[List] = None, 
    transform_coords:bool = True
) -> None:
    """
    Generates a cell interaction graph from a dataframe and save the json graph. The function reads
    x_centroid and y_centroid from dataframe file. By default its assuming that those values are in MPP

    Parameters:
    - df (DataFrame): input dataframe with x_centroid and y_centroid.
    - output_file (str): Path to the output pickle file where the graph will be saved.
    - interaction_radius_microns (float): Interaction radius in microns. Default value is 15
    - resolution_microns_per_pixel (float): Resolution of the imaging system in microns per pixel.
    - transcript_threshold (Optional[int]): Minimum transcript count to filter cells. Only used if provided.
    - sample_size (Optional[int]): Number of cells to sample for the graph. Only used if provided.
    - features:Optional[List]: Set of features to be used as node-level representation
    """

    # Filter cells based on transcript count threshold if provided
    if transcript_threshold is not None:
        df = df[df['transcript_counts'] >= transcript_threshold]

    # Randomly sample cells if sample_size is provided and there are more than 'sample_size' cells
    if sample_size is not None and df.shape[0] > sample_size:
        sampled_indices = np.random.choice(df.index, size=sample_size, replace=False)
        df = df.loc[sampled_indices]
    
    if transform_coords:
        # Convert cell coordinates to pixel space
        coordinates = df[['x_centroid', 'y_centroid']].to_numpy() / resolution_microns_per_pixel
        # Calculate the interaction radius in pixel space
        interaction_radius_pixels = interaction_radius_microns / resolution_microns_per_pixel
    else:
        coordinates = df[['x_centroid', 'y_centroid']].to_numpy().astype('int64')
        interaction_radius_pixels = interaction_radius_microns

    if linkage in ['Delaunay']:
        adjacency_matrix = adjacency_delaunay(coordinates,
                                            interaction_radius_pixels
                                            ).tocoo()
    else: # By defualt radius neighbor graph is used. 
        adjacency_matrix = radius_neighbors_graph(coordinates,
                                                interaction_radius_pixels,
                                                mode="connectivity",
                                                include_self=False).tocoo()
    # Get the edge indices
    edge_indices = np.vstack((adjacency_matrix.row, adjacency_matrix.col))

    # Prepare the graph dictionary
    graph_dict = {
        'pos': coordinates.tolist(),
        'edge_index': edge_indices.tolist(),
        'barcode': df['cell_id'].tolist(),
        #'IT_Region': df['IT_Region'].tolist()     
    }

    if features is not None:
        graph_dict['feats'] = df.loc[:, features].to_numpy().tolist()
        graph_dict['feat_names'] = features
    
    # Dropping some of the IT regions randomly. For testing the code set the TMP = True
    TMP = False
    if TMP:
        graph_dict = filter_graph_dict_by_random_region_removal(graph_dict,removal_fraction=0.5)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(graph_dict, f)
        print(f"Graph saved to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")


from torch_geometric.data import Data
import torch

def load_graph_from_json(json_path,FEATS_LIST=[]):
    with open(json_path, 'r') as f:
        graph_dict = json.load(f)

    import numpy as np
    if len(FEATS_LIST)>0:   # Feature section     
        graph_dict['feats'] = np.array(graph_dict['feats'])[:, np.isin(graph_dict['feat_names'], FEATS_LIST)]
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
    coordinates = torch.tensor(graph_dict['pos'], dtype=torch.float)

    # Create Data object
    data = Data(
        x=torch.tensor(graph_dict.get('feats'), dtype=torch.float) if 'feats' in graph_dict else None,
        edge_index=edge_index,
        pos=coordinates
    )

    # Add additional attributes
    data.barcode = graph_dict['barcode']
    if 'feat_names' in graph_dict:
        data.feat_names = graph_dict['feat_names']

    return data

def data_to_json(data: Data, scores: List[float] = []) -> Dict[str, Any]:
    """
    Convert a PyTorch Geometric Data object to a JSON-serializable dictionary.

    Parameters:
    ----------
    data : torch_geometric.data.Data
        The PyTorch Geometric Data object containing graph information.
    scores : list of float, optional
        A list of scores to be included in the JSON dictionary (default is an empty list).

    Returns:
    -------
    dict
        A dictionary representing the graph, suitable for JSON serialization.
    """
    graph_dict = {
        'coordinates': data.pos.tolist() if 'pos' in data else None,
        'edge_index': data.edge_index.tolist(),
        'barcode': data.barcode if 'barcode' in data else None,
        'IT_Region': data.IT_Region if 'IT_Region' in data else None
    }

    if 'x' in data:
        graph_dict['feats'] = data.x.tolist()
        graph_dict['feat_names'] = data.feat_names if 'feat_names' in data else None

    if 'y' in data:
        graph_dict['label'] = data.y.tolist()

    graph_dict['pid'] = data.pid
    graph_dict['score'] = scores

    return graph_dict


def create_geojson(df: pd.DataFrame, key='Selection', feature_type='Positive') -> dict:
    """
    Create GeoJSON FeatureCollection from DataFrame containing polygon data.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing polygon coordinates.
    - key (str): Column name in DataFrame representing unique selections (default: 'Selection').
    - feature_type (str): Type descriptor for the properties (default: 'Positive').

    Returns:
    - dict: GeoJSON FeatureCollection representation.
    """
    features = []
    
    for selection in df[key].unique():
        subset = df[df[key] == selection]
        coordinates = subset[['X', 'Y']].values.tolist()
        polygon = Polygon(coordinates)
        
        feature = {
            'type': 'Feature',
            'properties': {
                key: selection,
                'object_type': feature_type  # Correctly placing custom type in properties
            },
            'geometry': {
                'type': polygon.geom_type,
                'coordinates': [coordinates]  # Enclosing coordinates in an extra list for valid GeoJSON
            }
        }
        features.append(feature)
    
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    return geojson


def npy2pyramid(
    save_path: str,
    image: np.ndarray,
    metadata: str,
    mode: str = "rgb",
    channels: List[Dict[str, Union[str, int]]] = None,
    pyramid: dict = None,
    resolution: np.ndarray = None,
) -> None:
    """
    Saves a numpy array as a pyramid TIFF image with optional metadata.

    Args:
        save_path (str): The file path where the pyramid TIFF image will be saved.
        image (np.ndarray): The input image array. Can be a 2D (grayscale) or 3D (color or multiplex) numpy array.
        metadata (str): A string containing OME metadata for the image.
        mode (str): The mode of the image, either "rgb" for color images or "multiplex" for multiplex images. Defaults to "rgb".
        channels (list): A list of dictionaries where each dictionary contains metadata for each channel. Each dictionary must have keys that follow the OME-TIFF convention. The metadata for each channel corresponds to the same channel in the input image.
            Example:
            >>> channel_0 = {"color": (255, 0, 255), "name": "CD4"}
            >>> channel_1 = {"color": (0, 0, 255), "name": "CD8"}
            >>> channels = [channel_0, channel_1]
        pyramid (dict): A dictionary of keyword parameters for saving the image as a pyramid TIFF. If set to `None`, the following default parameters are used:
            >>> default_parameters = dict(
            >>>     compression="lzw",
            >>>     tile=True,
            >>>     tile_width=512,
            >>>     tile_height=512,
            >>>     pyramid=True,
            >>>     subifd=True,
            >>>     bigtiff=True,
            >>> )
        resolution (np.ndarray): An array containing the resolution in micrometers per pixel (um/pixel) for saving the input image. Should be a numpy array with two elements: [x_resolution, y_resolution].

    Raises:
        AssertionError: If the input image mode is not "rgb" or "multiplex".
        AssertionError: If the input image shape is not 2D or 3D.

    Examples:
        Save an RGB image as a pyramid TIFF:
        >>> npy2pyramid(save_path="output.tif", image=rgb_image, metadata=ome_metadata, mode="rgb")

        Save a multiplex image as a pyramid TIFF with channel metadata:
        >>> channels = [{"color": (255, 0, 255), "name": "CD4"}, {"color": (0, 0, 255), "name": "CD8"}]
        >>> npy2pyramid(save_path="output.tif", image=multiplex_image, metadata=ome_metadata, mode="multiplex", channels=channels)

    """
    
    # Will crash if the input image is smaller than the tile size
    default_pyramid = dict(
        compression="lzw",
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        subifd=True,
        bigtiff=True,
    )
    pyramid = default_pyramid if pyramid is None else pyramid

    np_dtype_to_vip_dtype = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }
    vi_dtype = np_dtype_to_vip_dtype[str(image.dtype)]

    image_shape = image.shape
    assert mode in ["rgb", "multiplex"]
    assert len(image_shape) in [2, 3]
    if len(image_shape) == 2:
        h, w = image_shape
        c = 1
    else:
        h, w, c = image_shape

    # `bands` is pyvips's terminology for channels
    image_ = image.reshape(h * w * c)
    vi = pyvips.Image.new_from_memory(image_.data, w, h, c, vi_dtype)
    im = vi
    if mode == "multiplex":
        # Reorganize as a toilet-roll image and format as OME-TIFF
        im = pyvips.Image.arrayjoin(vi.bandsplit(), across=1)

        # Set minimal OME metadata
        im = im.copy()
        channel_xml = []
        for ch_idx, ch_info in enumerate(channels):
            channel_xml.append(
                (
                    "<"
                    f"Channel ID='Channel:{ch_idx}' "
                    f"Color='{rgba_to_int(*ch_info['color'])}' "
                    f"Name='{ch_info['name']}' "
                    "/>"
                )
            )
        channel_xml = "\n".join(channel_xml)

    # Save the image with pyramid settings
    im.tiffsave(
        save_path,
        tile=True,
        pyramid=True,
        compression="lzw",
        Q=85,
        bigtiff=True,
        xres=1000 / resolution[0],
        yres=1000 / resolution[1],
        resunit="cm",
        tile_width=256,
        tile_height=256,
    )


def read_it_regions_annotations(
    directory_path: str,
    pattern: str = '*.csv'
    ) -> pd.DataFrame:
    """
    Combine CSV files in a directory matching a pattern and create a single dataframe.

    Args:
        directory_path (str): The directory path where the CSV files are located.
        pattern (str): The pattern to match CSV files. Default is '*.csv'.
        output_file (str): The path to save the combined DataFrame as a CSV file. Default is 'combined_dataframe.csv'.

    Returns:
        pd.DataFrame: The combined DataFrame with IT region assign to each cell. a cell belonging to multiple IT regions, to avoid duplication
        we drop duplicated entries. 
    """
    # Use glob to get all the CSV files matching the pattern
    csv_files = glob.glob(os.path.join(directory_path, pattern))
    
    # Initialize an empty list to hold the DataFrames
    dataframes = []

    # Read each CSV file and add it to the list
    for file in csv_files:
        df = pd.read_csv(file, skiprows=2)
        df['IT_Region'] = Path(file).stem.split('_')[-1]  # Add a column with the file name
        dataframes.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Assigning unique ID to each cell
    combined_df.index = combined_df['Cell ID']
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    # Return the combined dataframe
    return combined_df.loc[:,['IT_Region']]

def mkdirs(path):
    """
    Creates a directory and any necessary parent directories.
    If the directory already exists, it does nothing.

    Args:
        path (str): The path of the directory to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created successfully: {path}")
    except OSError as e:
        print(f"An error occurred while creating the directory: {e}")


