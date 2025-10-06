"""TMNG (Tau Monotonic Neighborhood Graph) implementation."""

from typing import Dict, List, Tuple, Callable, Any
import numpy as np
from .base import SearchGraph


def _process_tmng_batch(args: Tuple[List[int], Tuple[np.ndarray, np.ndarray, int, np.ndarray, float]]) -> List[Tuple[int, List[int]]]:
    """Process batch of nodes for TMNG construction."""
    batch_nodes, (NN, RevNN, n, distance_matrix, tau) = args
    results = []
    
    for u in batch_nodes:
        NN_u = NN[u]
        neighbors = []
        
        for rank in range(1, n):
            v = NN_u[rank]
            dist_uv = distance_matrix[u, v]
            
            blocked = False
            for w in neighbors:
                dist_wv = distance_matrix[w, v]
                if dist_wv < dist_uv - tau:
                    blocked = True
                    break
            
            if not blocked:
                neighbors.append(v)
        
        results.append((u, neighbors))
    
    return results


class TMNG(SearchGraph):
    """TMNG (Tau Monotonic Neighborhood Graph) implementation."""
    
    def __init__(self, distance_matrix: np.ndarray, neighbor_index=None, tau: float = 0.0):
        """Initialize TMNG."""
        super().__init__(distance_matrix, neighbor_index)
        self.tau = tau
    
    def _get_batch_processor(self) -> Callable:
        """Return batch processing function."""
        return _process_tmng_batch
    
    def _get_shared_data(self) -> Any:
        """Return shared data for batch processing."""
        NN, RevNN, n = self.neighbor_index.get_serializable_data()
        return (NN, RevNN, n, self.distance_matrix, self.tau)
    
