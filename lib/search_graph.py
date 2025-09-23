import numpy as np
from typing import List, Tuple, Dict, Set


class SearchGraph:
    """A directed graph with adjacency list representation for greedy ANN search."""
    
    def __init__(self, probability_matrix: np.ndarray, points: np.ndarray):
        """Initialize SearchGraph by sampling from probability matrix.
        
        Args:
            probability_matrix: Matrix of edge probabilities (n x n)
                               Binary matrices (0/1) are a special case
            points: Data points array (n x dim)
        """
        self.n = probability_matrix.shape[0]
        self.points = points
        
        # Sample adjacency matrix from probabilities
        # For binary matrices, this just returns the same matrix
        adjacency_matrix = np.random.binomial(1, probability_matrix)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
        
        # Convert to adjacency list for efficient neighbor lookup
        self.adjacency_list = {}
        for i in range(self.n):
            self.adjacency_list[i] = np.where(adjacency_matrix[i, :] == 1)[0].tolist()
    
    def greedy_search(self, start: int, query_vector: np.ndarray) -> int:
        """Perform greedy search from start towards query_vector.
        
        Args:
            start: Starting node index
            query_vector: Target query vector (arbitrary vector in same space)
            
        Returns:
            The node index where search terminates
        """
        current_node = start
        
        while True:
            # Get neighbors of current node
            neighbors = self.adjacency_list[current_node]
            
            if len(neighbors) == 0:
                # No outgoing edges, search terminates
                break
            
            # Find neighbor closest to query_vector (greedy choice)
            current_distance = np.linalg.norm(self.points[current_node] - query_vector)
            
            best_neighbor = None
            best_distance = float('inf')
            
            for neighbor in neighbors:
                distance = np.linalg.norm(self.points[neighbor] - query_vector)
                if distance < best_distance:
                    best_distance = distance
                    best_neighbor = neighbor
            
            # Only move if the best neighbor is closer than current node
            if best_neighbor is not None and best_distance < current_distance:
                current_node = best_neighbor
            else:
                # No progress possible, search terminates
                break
        
        return current_node