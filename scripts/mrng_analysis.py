"""MRNG sparsity analysis."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.mrng_builder import MRNGBuilder


if __name__ == "__main__":
    import numpy as np
    import time
    from lib.data_utils import load_fvecs, compute_distance_matrix
    
    print("MRNG Sparsity Analysis")
    print("=" * 50)
    
    # Load SIFT dataset
    print("Loading SIFT dataset...")
    sift_path = os.path.join(os.path.dirname(__file__), '..', 'siftsmall')
    vectors = load_fvecs(os.path.join(sift_path, 'siftsmall_base.fvecs'))
    print(f"   Loaded {len(vectors)} SIFT vectors of dimension {len(vectors[0])}")
    
    print(f"   Using complete dataset of {len(vectors)} points for MRNG")
    
    # Compute distance matrix
    print("Computing distance matrix...")
    start_time = time.time()
    distance_matrix = compute_distance_matrix(vectors)
    distance_time = time.time() - start_time
    print(f"   Distance matrix shape: {distance_matrix.shape}")
    print(f"   Distance computation took: {distance_time:.1f} seconds")
    
    # Build MRNG
    print("Building MRNG...")
    builder = MRNGBuilder(distance_matrix)
    
    # Analyze sparsity on sample
    sample_size = 1000
    print(f"Analyzing sparsity for {sample_size} sampled vertices...")
    print(f"   Dataset size: {len(vectors)} points in MRNG")
    print(f"   Sampling {sample_size} vertices to analyze their out-degrees")
    
    start_time = time.time()
    stats = builder.analyze_sparsity(sample_points=sample_size, random_seed=42)
    analysis_time = time.time() - start_time
    print(f"   MRNG analysis completed in {analysis_time:.1f} seconds")
    
    # Print results
    print("\nMRNG Sparsity Results:")
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
