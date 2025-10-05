from typing import Tuple
import numpy as np
import struct
from multiprocessing import Pool


def _compute_distance_row(args: Tuple[int, np.ndarray, np.ndarray]) -> Tuple[int, np.ndarray]:
    """Compute distances from one point to all others (vectorized)."""
    i, point_i, all_points = args
    distances = np.linalg.norm(all_points - point_i, axis=1)
    return i, distances


def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances using parallel processing."""
    n = len(points)
    distances = np.zeros((n, n))
    
    with Pool() as pool:
        args = [(i, points[i], points) for i in range(n)]
        results = pool.map(_compute_distance_row, args)
        
        for i, distance_row in results:
            distances[i] = distance_row
    
    return distances


def load_fvecs(filename: str) -> np.ndarray:
    """Load vectors from .fvecs format."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vector)
    return np.array(vectors)


def load_ivecs(filename: str) -> np.ndarray:
    """Load integer vectors from .ivecs format."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vector)
    return np.array(vectors)