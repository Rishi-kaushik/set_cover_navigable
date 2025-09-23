import numpy as np
from multiprocessing import Pool


def _process_point_neighbors_for_index(args):
    """Process a single point's neighbors for multiprocessing."""
    i, distance_row = args
    n = len(distance_row)
    
    # Get indices sorted by distance to point i
    distances_with_indices = [(distance_row[j], j) for j in range(n)]
    distances_with_indices.sort()
    
    nn_row = np.zeros(n, dtype=int)
    revnn_updates = {}
    
    for rank, (_, point_idx) in enumerate(distances_with_indices):
        nn_row[rank] = point_idx
        revnn_updates[point_idx] = rank
    
    return i, nn_row, revnn_updates


class NeighborIndex:
    def __init__(self, distance_matrix: np.ndarray):
        self.n = distance_matrix.shape[0]
        self.NN = np.zeros((self.n, self.n), dtype=int)  # NN[i,j] = jth closest point to i
        self.RevNN = np.zeros((self.n, self.n), dtype=int)  # RevNN[i,j] = rank of point j in NN[i]
        
        # Always use parallel processing
        with Pool() as pool:
            # Prepare arguments for each point
            args = [(i, distance_matrix[i]) for i in range(self.n)]
            
            # Process points in parallel
            results = pool.map(_process_point_neighbors_for_index, args)
            
            # Reconstruct matrices
            for i, nn_row, revnn_updates in results:
                self.NN[i] = nn_row
                for point_idx, rank in revnn_updates.items():
                    self.RevNN[i, point_idx] = rank
    
    def get_closer_points(self, target: int, reference: int) -> np.ndarray:
        """Get all points that are closer to target than reference point is.
        
        Args:
            target: The target point
            reference: The reference point to compare distances against
            
        Returns:
            Array of point indices that are closer to target than reference
        """
        reference_rank = self.RevNN[target, reference]
        return self.NN[target, :reference_rank]