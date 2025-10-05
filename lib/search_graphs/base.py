"""Base class for search graph implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple, Callable, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
from ..neighbor_index import NeighborIndex


class SearchGraph(ABC):
    """Base class for sparse navigable search graph implementations."""
    
    def __init__(self, distance_matrix: np.ndarray, neighbor_index: Optional[NeighborIndex] = None) -> None:
        """Initialize with distance matrix and optional precomputed neighbor index."""
        self.distance_matrix = distance_matrix
        self.neighbor_index = neighbor_index if neighbor_index is not None else NeighborIndex(distance_matrix)
        self.n = distance_matrix.shape[0]
    
    def build_graph(self, point_indices: List[int]) -> Dict[int, List[int]]:
        """Build graph using parallel batch processing."""
        return self._build_graph_parallel(point_indices)
    
    def _build_graph_parallel(self, point_indices: List[int]) -> Dict[int, List[int]]:
        """Parallel batch processing framework."""
        batches = self._create_batches(point_indices)
        process_func = self._get_batch_processor()
        shared_data = self._get_shared_data()
        
        batch_args = [(batch, shared_data) for batch in batches]
        
        with Pool() as pool:
            batch_results = pool.map(process_func, batch_args)
        
        return self._combine_batch_results(batch_results)
    
    def _create_batches(self, point_indices: List[int]) -> List[List[int]]:
        """Divide points into CPU-count batches."""
        num_processes = cpu_count()
        batch_size = len(point_indices) // num_processes
        if len(point_indices) % num_processes != 0:
            batch_size += 1
            
        return [
            point_indices[i:i + batch_size] 
            for i in range(0, len(point_indices), batch_size)
        ]
    
    def _combine_batch_results(self, batch_results: List[List[Tuple[int, List[int]]]]) -> Dict[int, List[int]]:
        """Combine batch results into graph dictionary."""
        graph = {}
        for batch_result in batch_results:
            for node, neighbors in batch_result:
                graph[node] = neighbors
        return graph
    
    @abstractmethod
    def _get_batch_processor(self) -> Callable:
        """Return batch processing function."""
        pass
    
    @abstractmethod 
    def _get_shared_data(self) -> Any:
        """Return shared data for batch processing."""
        pass
    
    def analyze_sparsity(self, point_indices: List[int]) -> Dict[str, Any]:
        """Analyze graph sparsity statistics."""
        graph = self.build_graph(point_indices)
        degrees = self._extract_degrees(graph, point_indices)
        
        if not degrees:
            degrees = []
        
        stats = {
            'num_points': len(point_indices),
            'total_edges': sum(degrees),
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'degree_distribution': degrees
        }
        
        return stats
    
    def _extract_degrees(self, graph: Dict[int, List[int]], 
                        points_to_analyze: List[int]) -> List[int]:
        """Extract degree list from graph."""
        degrees = []
        for i in points_to_analyze:
            try:
                degree = len(graph[i])
                degrees.append(degree)
            except (IndexError, KeyError):
                continue
        return degrees