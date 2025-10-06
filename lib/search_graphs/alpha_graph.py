"""Alpha Reachable Graph implementation."""

from typing import Dict, List, Tuple, Callable, Any
import numpy as np
from .base import SearchGraph


def _process_alpha_batch(args: Tuple[List[int], Tuple[np.ndarray, np.ndarray, int]]) -> List[Tuple[int, List[int]]]:
    """Process batch of nodes for Alpha Reachable Graph construction."""
    batch_nodes, (NN, RevNN, n) = args
    results = []
    
    for u in batch_nodes:
        NN_u = NN[u]
        neighbors = []
        
        for rank in range(1, n):
            v = NN_u[rank]
            
            blocked = False
            for w in neighbors:
                w_rank_from_v = RevNN[v, w]
                if w_rank_from_v < RevNN[v, u]:
                    blocked = True
                    break
            
            if not blocked:
                neighbors.append(v)
        
        results.append((u, neighbors))
    
    return results


class AlphaGraph(SearchGraph):
    """Alpha Reachable Graph implementation."""
    
    def _get_batch_processor(self) -> Callable:
        """Return batch processing function."""
        return _process_alpha_batch
    
    def _get_shared_data(self) -> Any:
        """Return shared data for batch processing."""
        return self.neighbor_index.get_serializable_data()
    
