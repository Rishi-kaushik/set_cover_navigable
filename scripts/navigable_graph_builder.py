"""Glue code for constructing sparse navigable graphs."""

import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.neighbor_index import NeighborIndex
from lib.set_cover import GreedySetCover


class NavigableGraphBuilder:
    """Constructs sparse navigable graphs using greedy set cover."""
    
    def __init__(self, neighbor_index: NeighborIndex):
        """Initialize with neighbor index."""
        self.neighbor_index = neighbor_index
        self.solver = GreedySetCover()
    
    def solve_for_point(self, point_i: int) -> list:
        """Find out-neighbors for point i using greedy set cover."""
        # Create set cover instance using neighbor index
        universe, sets_dict = self.neighbor_index.create_set_cover_instance(point_i)
        
        # Solve with greedy algorithm
        solution = self.solver.solve(universe, sets_dict)
        
        # Extract neighbor indices from solution set names
        neighbors = []
        for set_name in solution:
            # Parse "edge_i_to_k" format to get neighbor k
            parts = set_name.split('_')
            neighbor_k = int(parts[-1])  # Last part is the neighbor index
            neighbors.append(neighbor_k)
        
        return neighbors
    
    def build_navigable_graph(self, sample_points=None, random_seed=None):
        """Build navigable graph for sampled points."""
        n = self.neighbor_index.n
        
        if sample_points is not None:
            # Random sampling of points
            if random_seed is not None:
                random.seed(random_seed)
            points_to_analyze = random.sample(range(n), min(sample_points, n))
        else:
            # Analyze all points
            points_to_analyze = list(range(n))
        
        graph = {}
        total_points = len(points_to_analyze)
        for idx, i in enumerate(points_to_analyze):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"     Processing vertex {idx + 1}/{total_points} (point {i})...")
            graph[i] = self.solve_for_point(i)
        
        return graph
    
    def analyze_sparsity(self, sample_points=None, random_seed=None):
        """Analyze sparsity of the navigable graph."""
        graph = self.build_navigable_graph(sample_points, random_seed)
        
        degrees = [len(neighbors) for neighbors in graph.values()]
        
        stats = {
            'num_points': len(graph),
            'total_edges': sum(degrees),
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'degree_distribution': degrees
        }
        
        return stats


def quick_sparsity_analysis(distance_matrix, sample_points=None, random_seed=None):
    """Quick sparsity analysis from distance matrix."""
    neighbor_index = NeighborIndex(distance_matrix)
    builder = NavigableGraphBuilder(neighbor_index)
    return builder.analyze_sparsity(sample_points, random_seed)


if __name__ == "__main__":
    import numpy as np
    import time
    from lib.data_utils import load_fvecs, compute_distance_matrix
    
    print("Navigable Graph Sparsity Analysis")
    print("=" * 50)
    
    # Load SIFT dataset
    print("Loading SIFT dataset...")
    sift_path = os.path.join(os.path.dirname(__file__), '..', 'siftsmall')
    vectors = load_fvecs(os.path.join(sift_path, 'siftsmall_base.fvecs'))
    print(f"   Loaded {len(vectors)} SIFT vectors of dimension {len(vectors[0])}")
    
    # Use complete dataset for meaningful sparsity analysis
    print(f"   Using complete dataset of {len(vectors)} points for neighbor index")
    print(f"   Note: Building neighbor index for {len(vectors)} points will take time...")
    
    # Compute distance matrix
    print("Computing distance matrix...")
    start_time = time.time()
    distance_matrix = compute_distance_matrix(vectors)
    distance_time = time.time() - start_time
    print(f"   Distance matrix shape: {distance_matrix.shape}")
    print(f"   Distance computation took: {distance_time:.1f} seconds")
    
    # Build neighbor index (full cache needed for accurate set cover)
    print("Building neighbor index...")
    start_time = time.time()
    neighbor_index = NeighborIndex(distance_matrix)
    index_time = time.time() - start_time
    print(f"   Neighbor index built in {index_time:.1f} seconds")
    
    # Analyze sparsity: sample vertices, but each considers ALL points as neighbors
    sample_size = 1000  # Smaller sample since set cover on 10k points takes time
    print(f"Analyzing sparsity for {sample_size} sampled vertices...")
    print(f"   Dataset size: {len(vectors)} points in neighbor index")
    print(f"   Sampling {sample_size} vertices to analyze their out-degrees")
    print(f"   Each vertex can connect to any of the other {len(vectors)-1} points")
    print(f"   Note: Set cover solving on {len(vectors)} points will take time per vertex...")
    builder = NavigableGraphBuilder(neighbor_index)
    
    # Test one point to verify universe size
    sample_point = 0
    universe, sets_dict = neighbor_index.create_set_cover_instance(sample_point)
    print(f"   Verification: Point {sample_point} universe size = {len(universe)} points")
    print(f"   Verification: Point {sample_point} has {len(sets_dict)} potential neighbors")
    
    start_time = time.time()
    stats = builder.analyze_sparsity(sample_points=sample_size, random_seed=42)
    analysis_time = time.time() - start_time
    print(f"   Sparsity analysis completed in {analysis_time:.1f} seconds")
    
    # Print results
    print("\nSparsity Analysis Results:")
    print(f"   Sample size: {stats['num_points']} points")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Average degree: {stats['avg_degree']:.2f}")
    print(f"   Max degree: {stats['max_degree']}")
    print(f"   Min degree: {stats['min_degree']}")
    
    # Compare to theoretical bound
    import math
    n = len(vectors)
    theoretical_bound = math.log(n)
    print(f"\nComparison to Theory:")
    print(f"   Dataset size (n): {n}")
    print(f"   Theoretical O(log n) bound: {theoretical_bound:.2f}")
    print(f"   Actual average degree: {stats['avg_degree']:.2f}")
    print(f"   Ratio (actual/theory): {stats['avg_degree']/theoretical_bound:.2f}x")
    
    # Degree distribution summary
    degrees = stats['degree_distribution']
    print(f"\nDegree Distribution:")
    print(f"   25th percentile: {np.percentile(degrees, 25):.1f}")
    print(f"   50th percentile: {np.percentile(degrees, 50):.1f}")
    print(f"   75th percentile: {np.percentile(degrees, 75):.1f}")
    print(f"   95th percentile: {np.percentile(degrees, 95):.1f}")
    
    print("\nAnalysis complete!")
