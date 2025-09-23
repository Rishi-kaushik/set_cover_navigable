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
    
    def build_navigable_graph(self, sample_points: int | None = None, random_seed: int | None = None) -> dict:
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
        for i in points_to_analyze:
            graph[i] = self.solve_for_point(i)
        
        return graph
    
    def analyze_sparsity(self, sample_points: int | None = None, random_seed: int | None = None) -> dict:
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


def quick_sparsity_analysis(distance_matrix, sample_points: int | None = None, random_seed: int | None = None) -> dict:
    """Quick sparsity analysis from distance matrix."""
    neighbor_index = NeighborIndex(distance_matrix)
    builder = NavigableGraphBuilder(neighbor_index)
    return builder.analyze_sparsity(sample_points, random_seed)


if __name__ == "__main__":
    import numpy as np
    from lib.data_utils import load_fvecs, compute_distance_matrix
    
    print("🔬 Navigable Graph Sparsity Analysis")
    print("=" * 50)
    
    # Load SIFT dataset
    print("📊 Loading SIFT dataset...")
    sift_path = os.path.join(os.path.dirname(__file__), '..', 'siftsmall')
    vectors = load_fvecs(os.path.join(sift_path, 'siftsmall_base.fvecs'))
    print(f"   Loaded {len(vectors)} SIFT vectors of dimension {len(vectors[0])}")
    
    # Use subset for faster computation
    subset_size = 1000  # Use 1000 points for neighbor index
    if len(vectors) > subset_size:
        vectors = vectors[:subset_size]
        print(f"   Using subset of {len(vectors)} points for analysis")
    
    # Compute distance matrix
    print("📏 Computing distance matrix...")
    distance_matrix = compute_distance_matrix(vectors)
    print(f"   Distance matrix shape: {distance_matrix.shape}")
    
    # Build neighbor index (full cache needed for accurate set cover)
    print("🏗️  Building neighbor index...")
    neighbor_index = NeighborIndex(distance_matrix)
    print("   ✓ Neighbor index built")
    
    # Analyze sparsity on sample of 100 points
    print("🎯 Analyzing sparsity for sample of 100 points...")
    builder = NavigableGraphBuilder(neighbor_index)
    stats = builder.analyze_sparsity(sample_points=100, random_seed=42)
    
    # Print results
    print("\n📈 Sparsity Analysis Results:")
    print(f"   Sample size: {stats['num_points']} points")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Average degree: {stats['avg_degree']:.2f}")
    print(f"   Max degree: {stats['max_degree']}")
    print(f"   Min degree: {stats['min_degree']}")
    
    # Compare to theoretical bound
    import math
    n = len(vectors)
    theoretical_bound = math.log(n)
    print(f"\n🎯 Comparison to Theory:")
    print(f"   Dataset size (n): {n}")
    print(f"   Theoretical O(log n) bound: {theoretical_bound:.2f}")
    print(f"   Actual average degree: {stats['avg_degree']:.2f}")
    print(f"   Ratio (actual/theory): {stats['avg_degree']/theoretical_bound:.2f}x")
    
    # Degree distribution summary
    degrees = stats['degree_distribution']
    print(f"\n📊 Degree Distribution:")
    print(f"   25th percentile: {np.percentile(degrees, 25):.1f}")
    print(f"   50th percentile: {np.percentile(degrees, 50):.1f}")
    print(f"   75th percentile: {np.percentile(degrees, 75):.1f}")
    print(f"   95th percentile: {np.percentile(degrees, 95):.1f}")
    
    print("\n✅ Analysis complete!")
