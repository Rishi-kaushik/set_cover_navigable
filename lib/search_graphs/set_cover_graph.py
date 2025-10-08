"""Set cover graph construction using greedy set cover algorithm optimized with reverse index."""

from typing import List, Any, Tuple, Callable
import numpy as np
from .base import SearchGraph


def _greedy_set_cover_optimized(point_i: int, NN: np.ndarray, RevNN: np.ndarray, n: int) -> List[int]:
    """Greedy set cover using reverse index directly without creating explicit sets."""
    if n <= 1:
        return []
    
    uncovered = np.ones(n, dtype=bool)
    uncovered[point_i] = False
    neighbors = []
    
    num_uncovered = n - 1
    
    candidate_neighbors = np.arange(n)
    candidate_mask = np.ones(n, dtype=bool)
    candidate_mask[point_i] = False
    
    while num_uncovered > 0:
        best_neighbor = -1
        max_coverage = 0
        
        for k in candidate_neighbors[candidate_mask]:
            rank_of_i_from_k = RevNN[k, point_i]
            
            if rank_of_i_from_k == 0:
                candidate_mask[k] = False
                continue
            
            if rank_of_i_from_k <= max_coverage:
                continue
            
            closer_points = NN[k, :rank_of_i_from_k]
            coverage = np.sum(uncovered[closer_points])
            
            if coverage > max_coverage:
                max_coverage = coverage
                best_neighbor = k
        
        if best_neighbor == -1 or max_coverage == 0:
            break
        
        neighbors.append(best_neighbor)
        candidate_mask[best_neighbor] = False
        
        rank_of_i_from_best = RevNN[best_neighbor, point_i]
        covered_points = NN[best_neighbor, :rank_of_i_from_best]
        uncovered[covered_points] = False
        num_uncovered = np.sum(uncovered)
    
    return neighbors


def _process_setcover_batch(args: Tuple[List[int], Tuple[np.ndarray, np.ndarray, int]]) -> List[Tuple[int, List[int]]]:
    """Process batch of nodes for set cover graph construction."""
    batch_nodes, (NN, RevNN, n) = args
    results = []
    
    for point_i in batch_nodes:
        neighbors = _greedy_set_cover_optimized(point_i, NN, RevNN, n)
        results.append((point_i, neighbors))
    
    return results


class SetCoverGraph(SearchGraph):
    """Set cover based navigable graph."""
    
    def _get_batch_processor(self) -> Callable:
        """Return batch processing function."""
        return _process_setcover_batch
    
    def _get_shared_data(self) -> Any:
        """Return shared data for batch processing."""
        return self.neighbor_index.get_serializable_data()
