from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon
import numpy as np

def calculate_points_density(points, radius):
    """Calculate the local density of points in a given radius using a KDTree."""
    tree = KDTree(points)
    densities = tree.query_radius(points, r=radius, count_only=True)
    return densities

def alpha_shape(points, alpha):
    """Compute the alpha shape (concave hull) of a set of points."""
    if len(points) < 4:
        return Polygon(points)
    
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i, j in zip(simplex, np.roll(simplex, -1)):
            edges.add(tuple(sorted([i, j])))
    
    edge_points = np.array([points[i] for i, j in edges if np.linalg.norm(points[i] - points[j]) < alpha])
    if len(edge_points) < 3:
        return None
    return Polygon(edge_points).convex_hull