#!/usr/bin/env python3
"""Analyze sparsity of different graph construction algorithms on SIFT dataset.

Configuration:
  - use_full_dataset: True = use all dataset points, False = sample subset
  - dataset_sample_size: Number of points to sample (if use_full_dataset=False)
  - edge_compute_size: Number of points to compute outgoing edges for
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from lib.data_utils import load_fvecs, compute_distance_matrix
from lib.search_graphs import MRNG, SetCoverGraph, AlphaGraph, TMNG
from lib.neighbor_index import NeighborIndex


def main():
    # ============ CONFIGURATION ============
    use_full_dataset = False    # True = use all dataset points, False = sample subset
    dataset_sample_size = 2000  # Only used if use_full_dataset = False
    edge_compute_size = 500     # Number of points to compute edges for
    # ======================================
    
    print("=" * 80)
    print(f"Graph Sparsity Analysis on SIFT Dataset")
    print("=" * 80)
    
    print("\n[1/4] Loading SIFT dataset...")
    start_time = time.time()
    base_vectors = load_fvecs('siftsmall/siftsmall_base.fvecs')
    load_time = time.time() - start_time
    total_points = len(base_vectors)
    
    if use_full_dataset:
        all_points = base_vectors
        dataset_size = total_points
        print(f"  Loaded {total_points} vectors of dimension {base_vectors.shape[1]}")
        print(f"  Using FULL dataset")
    else:
        all_points = base_vectors[:dataset_sample_size]
        dataset_size = dataset_sample_size
        print(f"  Loaded {total_points} vectors of dimension {base_vectors.shape[1]}")
        print(f"  Sampling {dataset_sample_size} points from dataset")
    
    print(f"  Time: {load_time:.2f}s")
    
    print("\n" + "=" * 80)
    print(f"Dataset size:    {total_points} points (full)")
    print(f"Working dataset: {dataset_size} points")
    print(f"Computing edges: {edge_compute_size} points")
    print(f"Edge targets:    All {dataset_size} points in working dataset")
    print("=" * 80)
    
    print("\n[2/4] Computing distance matrix...")
    print(f"  Computing distances for {dataset_size} points...")
    start_time = time.time()
    distance_matrix = compute_distance_matrix(all_points)
    distance_time = time.time() - start_time
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    print(f"  Time: {distance_time:.2f}s")
    
    print("\n[3/4] Computing shared NeighborIndex (reused across all graphs)...")
    start_time = time.time()
    neighbor_index = NeighborIndex(distance_matrix)
    neighbor_index_time = time.time() - start_time
    print(f"  NeighborIndex computed for {neighbor_index.n} points")
    print(f"  Time: {neighbor_index_time:.2f}s")
    
    edge_point_indices = list(range(edge_compute_size))
    
    graph_configs = [
        ('MRNG', MRNG, {}),
        ('SetCoverGraph', SetCoverGraph, {}),
        ('AlphaGraph (α=1.0)', AlphaGraph, {'alpha': 1.0}),
        ('AlphaGraph (α=2.0)', AlphaGraph, {'alpha': 2.0}),
        ('TMNG (τ=0.0)', TMNG, {'tau': 0.0}),
        ('TMNG (τ=0.1)', TMNG, {'tau': 0.1}),
    ]
    
    print("\n[4/4] Building graphs and analyzing sparsity...")
    print("=" * 80)
    print(f"Computing outgoing edges for {edge_compute_size} points")
    print(f"Each point can connect to any of {dataset_size} points in working dataset")
    print("=" * 80)
    
    results = {}
    
    for graph_name, graph_class, graph_params in graph_configs:
        print(f"\n{graph_name}:")
        print("-" * 40)
        
        print(f"  Initializing with {dataset_size} points...")
        graph_builder = graph_class(distance_matrix, neighbor_index, **graph_params)
        
        print(f"  Computing edges for {edge_compute_size} points...")
        start_time = time.time()
        stats = graph_builder.analyze_sparsity(edge_point_indices)
        graph_time = time.time() - start_time
        
        stats['runtime'] = graph_time
        results[graph_name] = stats
        
        print(f"  Sampled points:      {stats['num_points']}")
        print(f"  Total edges:         {stats['total_edges']}")
        print(f"  Average degree:      {stats['avg_degree']:.2f}")
        print(f"  Max degree:          {stats['max_degree']}")
        print(f"  Min degree:          {stats['min_degree']}")
        print(f"  Runtime:             {graph_time:.2f}s")
    
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"{'Graph Type':<20} {'Avg Degree':<15} {'Total Edges':<15} {'Runtime (s)':<15}")
    print("-" * 80)
    
    for graph_name, stats in results.items():
        print(f"{graph_name:<20} {stats['avg_degree']:<15.2f} {stats['total_edges']:<15} {stats['runtime']:<15.2f}")
    
    print("\n" + "=" * 80)
    print("Runtime Breakdown")
    print("=" * 80)
    print(f"Loading dataset:              {load_time:.2f}s")
    print(f"Computing distance matrix:    {distance_time:.2f}s")
    print(f"Computing NeighborIndex:      {neighbor_index_time:.2f}s")
    for graph_name, stats in results.items():
        print(f"{graph_name + ' graph construction:':<30} {stats['runtime']:.2f}s")
    
    total_time = load_time + distance_time + neighbor_index_time + sum(s['runtime'] for s in results.values())
    print(f"{'Total:':<30} {total_time:.2f}s")
    
    print("\n" + "=" * 80)
    print(f"Note: Computed edges for {edge_compute_size} points from a")
    print(f"working dataset of {dataset_size} points. Each point can connect to any")
    print(f"of the {dataset_size} points in the working dataset.")
    print("=" * 80)
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
