"""Set cover graph construction using greedy set cover algorithm."""

from typing import Set, Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from .base import SearchGraph


class GreedySetCover:
    """Greedy O(log n) set cover approximation algorithm."""
    
    def solve(self, universe: Set[int], sets_dict: Dict[str, Set[int]]) -> List[str]:
        """Solve set cover greedily."""
        if not universe or not sets_dict:
            return []
        
        universe = set(universe)
        sets_dict = {name: set(elements) for name, elements in sets_dict.items()}
        
        uncovered = universe.copy()
        solution = []
        available_sets = sets_dict.copy()
        
        while uncovered and available_sets:
            best_set_name = self._find_best_set(uncovered, available_sets)
            
            if best_set_name is None:
                break
            
            solution.append(best_set_name)
            uncovered -= available_sets[best_set_name]
            del available_sets[best_set_name]
        
        return solution
    
    def _find_best_set(self, uncovered: Set[int], 
                       available_sets: Dict[str, Set[int]]) -> Optional[str]:
        """Find set covering most uncovered elements."""
        best_set_name = None
        max_coverage = 0
        
        for set_name, set_elements in available_sets.items():
            coverage = len(set_elements & uncovered)
            
            if coverage > max_coverage:
                max_coverage = coverage
                best_set_name = set_name
        
        return best_set_name


def _get_set_elements_from_data(point_i: int, neighbor_k: int, NN: np.ndarray, RevNN: np.ndarray) -> Set[int]:
    """Get points closer to neighbor_k than to point_i."""
    if point_i == neighbor_k:
        return set()
    
    rank_of_i_from_k = RevNN[neighbor_k, point_i]
    closer_to_k = NN[neighbor_k, :rank_of_i_from_k]
    elements = set(closer_to_k) - {point_i}
    
    return elements


def _create_set_cover_instance_from_data(point_i: int, NN: np.ndarray, RevNN: np.ndarray, n: int) -> Tuple[Set[int], Dict[str, Set[int]]]:
    """Create set cover instance for point_i."""
    universe = set(range(n)) - {point_i}
    
    sets_dict = {}
    for k in range(n):
        if k != point_i:
            set_elements = _get_set_elements_from_data(point_i, k, NN, RevNN)
            sets_dict[f'edge_{point_i}_to_{k}'] = set_elements
    
    return universe, sets_dict


def _process_setcover_batch(args: Tuple[List[int], Tuple[np.ndarray, np.ndarray, int]]) -> List[Tuple[int, List[int]]]:
    """Process batch of nodes for set cover graph construction."""
    batch_nodes, (NN, RevNN, n) = args
    solver = GreedySetCover()
    results = []
    
    for point_i in batch_nodes:
        universe, sets_dict = _create_set_cover_instance_from_data(point_i, NN, RevNN, n)
        solution = solver.solve(universe, sets_dict)
        
        neighbors = []
        for set_name in solution:
            parts = set_name.split('_')
            neighbor_k = int(parts[-1])
            neighbors.append(neighbor_k)
        
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
    


