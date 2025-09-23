"""Test the NavigableGraphBuilder glue code."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scripts.navigable_graph_builder import NavigableGraphBuilder, quick_sparsity_analysis
from lib.neighbor_index import NeighborIndex


def test_navigable_graph_builder():
    """Test NavigableGraphBuilder with small synthetic data."""
    print("Testing NavigableGraphBuilder...")
    
    # Create small test distance matrix (5x5)
    np.random.seed(42)
    points = np.random.rand(5, 2)  # 5 points in 2D
    
    # Compute distance matrix
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    
    print(f"Created {n}x{n} distance matrix")
    
    # Test the builder
    neighbor_index = NeighborIndex(distance_matrix)
    builder = NavigableGraphBuilder(neighbor_index)
    
    # Test single point solution
    neighbors = builder.solve_for_point(0)
    print(f"Point 0 out-neighbors: {neighbors}")
    
    # Test full graph construction
    graph = builder.build_navigable_graph()
    print(f"Complete graph: {graph}")
    
    # Test sparsity analysis
    stats = builder.analyze_sparsity()
    print(f"Sparsity stats: {stats}")
    
    # Test convenience function
    quick_stats = quick_sparsity_analysis(distance_matrix)
    print(f"Quick analysis: {quick_stats}")
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_navigable_graph_builder()
