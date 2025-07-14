import json
import os
import multiprocessing as mp
from typing import List
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance
from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from application.project_cp import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir
from application.utils.shape_approx import calculate_points_density, alpha_shape

def process_sample(sample_id, df, output_dir, radius=500, density_threshold=50, alpha=None, inner_strip_width=100):
    centroids = df[df['sample_key'] == sample_id][['x', 'y']].to_numpy()
    if centroids.size == 0:
        print(f"No data found for sample {sample_id}. Skipping.")
        return

    # Compute point densities and filter by threshold
    densities = calculate_points_density(centroids, radius)
    filtered_centroids = centroids[densities >= density_threshold]

    # If alpha is None, skip concave hull and plot based on density only
    if alpha is None:
        # Single plot: all points and high-density points
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=1, alpha=0.5, label='All Cells')
        ax.scatter(filtered_centroids[:, 0], filtered_centroids[:, 1], c='red', s=1, alpha=0.5, label='High Density')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Density Threshold Plot for sample {sample_id}")
        ax.legend()
        ax.axis("equal")

        # Save the figure
        mkdir(output_dir)
        output_path = os.path.join(output_dir, f"density_only_{sample_id}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved Density plot for sample {sample_id} at {output_path}")
        return

    # Ensure enough points for concave hull when alpha is provided
    if len(filtered_centroids) < 3:
        print(f"Not enough points to compute concave hull in {sample_id}. Skipping.")
        return

    # Compute concave hull
    concave_hull = alpha_shape(filtered_centroids, alpha)
    if concave_hull is None:
        print(f"Concave hull computation failed in {sample_id}. Skipping.")
        return

    # Create inner boundary
    inner_boundary = concave_hull.buffer(-inner_strip_width)
    if inner_boundary.is_empty:
        print(f"Inner boundary is empty for cells in {sample_id}. Skipping.")
        return

    # Determine points inside the hulls
    inside_hull = np.array([concave_hull.contains(Point(p)) for p in centroids])
    inside_inner = np.array([inner_boundary.contains(Point(p)) for p in centroids])
    points_inside_hull = centroids[inside_hull]
    points_outside_hull = centroids[~inside_hull]
    points_inside_inner = centroids[inside_inner]

    # Plot original points, concave hull, and density selection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Points and concave hull
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=1, alpha=0.5, label='All Cells')
    ax1.scatter(filtered_centroids[:, 0], filtered_centroids[:, 1], c='red', s=1, alpha=0.5, label='High Density')
    x, y = concave_hull.exterior.xy
    ax1.plot(x, y, 'r--', lw=2, label='Concave Hull')
    ax1.fill(x, y, 'r', alpha=0.1)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.set_title(f"Concave Hull for cells in Sample {sample_id}")
    ax1.legend()
    ax1.axis("equal")

    # Subplot 2: Inside and outside hull, inner strip
    ax2.scatter(points_outside_hull[:, 0], points_outside_hull[:, 1], c='blue', s=1, alpha=0.5, label='Outside Hull')
    ax2.scatter(points_inside_hull[:, 0], points_inside_hull[:, 1], c='green', s=1, alpha=0.5, label='Inside Hull')
    ax2.scatter(points_inside_inner[:, 0], points_inside_inner[:, 1], c='orange', s=1, alpha=0.5, label='Inside Inner Strip')

    x, y = concave_hull.exterior.xy
    ax2.plot(x, y, 'r--', lw=2, label='Concave Hull')
    ax2.fill(x, y, 'r', alpha=0.1)
    x_inner, y_inner = inner_boundary.exterior.xy
    ax2.plot(x_inner, y_inner, 'g--', lw=2, label='Inner Boundary')
    ax2.fill(x_inner, y_inner, 'g', alpha=0.1)

    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.set_title(f"Points Inside Concave Hull and Inner Strip (Width = {inner_strip_width})")
    ax2.legend()
    ax2.axis("equal")

    # Save the figure
    mkdir(output_dir)
    output_path = os.path.join(output_dir, f"density_concave_hull_{sample_id}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Density + Concave Hull + Inner Strip plot for cells in {sample_id} at {output_path}")

def main():
    CONSOL_FILE = f'{DATA_DIR}/cell_paper_data.csv'
    consolDf = pd.read_csv(CONSOL_FILE)

    selected_columns = ['sample_key', 'cell_type', 'x', 'y', 'cell_id']
    mapping = dict(
        zip(
            ['H_field', ANNOT_LEVEL, 'x.coord', 'y.coord', 'Unnamed: 0'],
            selected_columns
        )
    )
    consolDf.rename(columns=mapping, inplace=True)

    cases = list(set(consolDf['sample_key']))

    for sample_id in cases:
        OUT_DIR = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/SelectedCells/'
        process_sample(sample_id, consolDf, OUT_DIR, alpha=500)

if __name__ == "__main__":
    main()