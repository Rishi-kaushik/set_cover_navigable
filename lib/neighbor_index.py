from typing import Tuple, List, Dict, Set
import numpy as np
from multiprocessing import Pool


def _process_point_neighbors_for_index(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, Dict[int, int]]:
    """Process point's neighbors for parallel index construction."""
    i, distance_row = args
    n = len(distance_row)
    
    distances_with_indices = [(distance_row[j], j) for j in range(n)]
    distances_with_indices.sort()
    
    nn_row = np.zeros(n, dtype=int)
    revnn_updates = {}
    
    for rank, (_, point_idx) in enumerate(distances_with_indices):
        nn_row[rank] = point_idx
        revnn_updates[point_idx] = rank
    
    return i, nn_row, revnn_updates


class NeighborIndex:
    def __init__(self, distance_matrix: np.ndarray) -> None:
        self.n = distance_matrix.shape[0]
        self.NN = np.zeros((self.n, self.n), dtype=int)
        self.RevNN = np.zeros((self.n, self.n), dtype=int)
        
        with Pool() as pool:
            args = [(i, distance_matrix[i]) for i in range(self.n)]
            results = pool.map(_process_point_neighbors_for_index, args)
            
            for i, nn_row, revnn_updates in results:
                self.NN[i] = nn_row
                for point_idx, rank in revnn_updates.items():
                    self.RevNN[i, point_idx] = rank
    
    def get_closer_points(self, target: int, reference: int) -> np.ndarray:
        """Get points closer to target than reference."""
        reference_rank = self.RevNN[target, reference]
        return self.NN[target, :reference_rank]
    
    def get_set_elements(self, point_i: int, neighbor_k: int) -> Set[int]:
        """Get points closer to neighbor_k than to point_i."""
        if point_i == neighbor_k:
            return set()
        
        rank_of_i_from_k = self.RevNN[neighbor_k, point_i]
        closer_to_k = self.NN[neighbor_k, :rank_of_i_from_k]
        elements = set(closer_to_k) - {point_i}
        
        return elements
    
    def create_set_cover_instance(self, point_i: int) -> Tuple[Set[int], Dict[str, Set[int]]]:
        """Create set cover instance for point_i."""
        universe = set(range(self.n)) - {point_i}
        
        sets_dict = {}
        for k in range(self.n):
            if k != point_i:
                set_elements = self.get_set_elements(point_i, k)
                sets_dict[f'edge_{point_i}_to_{k}'] = set_elements
        
        return universe, sets_dict
    
    def get_serializable_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return (NN, RevNN, n) for multiprocessing."""
        return (self.NN, self.RevNN, self.n)
    
