"""Compare Set Cover vs MRNG sparsity."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.neighbor_index import NeighborIndex
from lib.mrng_builder import MRNGBuilder
from scripts.navigable_graph_builder import NavigableGraphBuilder


if __name__ == "__main__":
    import numpy as np
    import time
    from lib.data_utils import load_fvecs, compute_distance_matrix
    
    print("Set Cover vs MRNG Comparison")
    print("=" * 50)
    
    # Load SIFT dataset
    print("Loading SIFT dataset...")
    sift_path = os.path.join(os.path.dirname(__file__), '..', 'siftsmall')
    vectors = load_fvecs(os.path.join(sift_path, 'siftsmall_base.fvecs'))
    print(f"   Loaded {len(vectors)} SIFT vectors of dimension {len(vectors[0])}")
    
    # Compute distance matrix
    print("Computing distance matrix...")
    start_time = time.time()
    distance_matrix = compute_distance_matrix(vectors)
    distance_time = time.time() - start_time
    print(f"   Distance computation took: {distance_time:.1f} seconds")
    
    sample_size = 50
    print(f"\nAnalyzing {sample_size} sampled vertices for both methods...")
    
    # Build shared neighbor index
    print("\nBuilding shared neighbor index...")
    start_time = time.time()
    neighbor_index = NeighborIndex(distance_matrix)
    index_time = time.time() - start_time
    print(f"   Neighbor index built in {index_time:.1f} seconds")
    
    # MRNG Analysis
    print("\n1. MRNG Analysis:")
    mrng_builder = MRNGBuilder(distance_matrix)  # Will reuse neighbor index internally
    start_time = time.time()
    mrng_stats = mrng_builder.analyze_sparsity(sample_points=sample_size, random_seed=42)
    mrng_time = time.time() - start_time
    print(f"   MRNG completed in {mrng_time:.1f} seconds")
    
    # Set Cover Analysis
    print("\n2. Set Cover Analysis:")
    sc_builder = NavigableGraphBuilder(neighbor_index)
    start_time = time.time()
    sc_stats = sc_builder.analyze_sparsity(sample_points=sample_size, random_seed=42)
    sc_time = time.time() - start_time
    print(f"   Set Cover completed in {sc_time:.1f} seconds")
    
    # Comparison table
    print("\n" + "=" * 60)
    print("SPARSITY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'MRNG':<15} {'Set Cover':<15} {'Ratio':<10}")
    print("-" * 60)
    print(f"{'Avg Degree':<20} {mrng_stats['avg_degree']:<15.2f} {sc_stats['avg_degree']:<15.2f} {mrng_stats['avg_degree']/sc_stats['avg_degree']:<10.2f}")
    print(f"{'Max Degree':<20} {mrng_stats['max_degree']:<15} {sc_stats['max_degree']:<15} {mrng_stats['max_degree']/sc_stats['max_degree']:<10.2f}")
    print(f"{'Min Degree':<20} {mrng_stats['min_degree']:<15} {sc_stats['min_degree']:<15} {mrng_stats['min_degree']/sc_stats['min_degree']:<10.2f}")
    print(f"{'Total Edges':<20} {mrng_stats['total_edges']:<15} {sc_stats['total_edges']:<15} {mrng_stats['total_edges']/sc_stats['total_edges']:<10.2f}")
    
    # Timing comparison
    print(f"\n{'Construction Time':<20} {mrng_time:<15.1f} {sc_time:<15.1f} {mrng_time/sc_time:<10.2f}")
    
    # Theoretical comparison
    import math
    theoretical_bound = math.log(len(vectors))
    print(f"\nComparison to O(log n) = {theoretical_bound:.2f}:")
    print(f"   MRNG ratio: {mrng_stats['avg_degree']/theoretical_bound:.2f}x")
    print(f"   Set Cover ratio: {sc_stats['avg_degree']/theoretical_bound:.2f}x")
    
    print("\nComparison complete!")