"""MRNG (Monotonic Relative Neighborhood Graph) construction."""

import numpy as np
from .neighbor_index import NeighborIndex


def build_mrng_with_index(neighbor_index):
    """Build MRNG using precomputed neighbor index."""
    n = neighbor_index.n
    adjacency_list = [[] for _ in range(n)]
    
    for u in range(n):
        # Iterate through points sorted by distance from u
        for rank in range(1, n):  # Skip rank 0 (u itself)
            v = neighbor_index.NN[u, rank]
            
            # Check if edge (u,v) should be added
            blocked = False
            for w in adjacency_list[u]:  # Existing neighbors of u
                w_rank_from_v = neighbor_index.RevNN[v, w]
                if w_rank_from_v < neighbor_index.RevNN[v, u]:
                    blocked = True
                    break
            
            if not blocked:
                adjacency_list[u].append(v)
    
    return adjacency_list


class MRNGBuilder:
    """Build and analyze MRNG graphs."""
    
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.neighbor_index = NeighborIndex(distance_matrix)
        self.n = distance_matrix.shape[0]
    
    def build_mrng(self):
        """Build MRNG graph."""
        return build_mrng_with_index(self.neighbor_index)
    
    def analyze_sparsity(self, sample_points=None, random_seed=None):
        """Analyze sparsity of MRNG graph."""
        import random
        
        if sample_points is not None:
            if random_seed is not None:
                random.seed(random_seed)
            points_to_analyze = random.sample(range(self.n), min(sample_points, self.n))
        else:
            points_to_analyze = list(range(self.n))
        
        # Build full MRNG
        adjacency_list = self.build_mrng()
        
        # Analyze sampled points
        degrees = []
        total_edges = 0
        for i in points_to_analyze:
            degree = len(adjacency_list[i])
            degrees.append(degree)
            total_edges += degree
        
        stats = {
            'num_points': len(points_to_analyze),
            'total_edges': total_edges,
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'degree_distribution': degrees
        }
        
        return stats
