#!/usr/bin/env python3
"""Analyze sparsity of different graph construction algorithms on SIFT dataset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from lib.data_utils import load_fvecs, compute_distance_matrix
from lib.search_graphs import MRNG, SetCoverGraph, AlphaGraph, TMNG
from lib.neighbor_index import NeighborIndex


def main():
    sample_size = 500
    
    print("=" * 80)
    print(f"Graph Sparsity Analysis on SIFT Dataset")
    print("=" * 80)
    
    print("\n[1/4] Loading SIFT dataset...")
    start_time = time.time()
    base_vectors = load_fvecs('siftsmall/siftsmall_base.fvecs')
    load_time = time.time() - start_time
    total_points = len(base_vectors)
    
    print(f"  Loaded {total_points} vectors of dimension {base_vectors.shape[1]}")
    print(f"  Time: {load_time:.2f}s")
    
    print("\n" + "=" * 80)
    print(f"Dataset size:    {total_points} points")
    print(f"Sample size:     {sample_size} points (computing outgoing edges)")
    print(f"Edge targets:    All {total_points} points in dataset")
    print("=" * 80)
    
    all_points = base_vectors
    
    print("\n[2/4] Computing distance matrix...")
    print(f"  Computing distances for all {total_points} points...")
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
    
    sampled_point_indices = list(range(sample_size))
    
    graph_types = {
        'MRNG': MRNG,
        'SetCoverGraph': SetCoverGraph,
        'AlphaGraph': AlphaGraph,
        'TMNG': TMNG
    }
    
    print("\n[4/4] Building graphs and analyzing sparsity...")
    print("=" * 80)
    print(f"Computing outgoing edges for {sample_size} sampled points")
    print(f"Each sampled point can connect to any of {total_points} points")
    print("=" * 80)
    
    results = {}
    
    for graph_name, graph_class in graph_types.items():
        print(f"\n{graph_name}:")
        print("-" * 40)
        
        print(f"  Initializing with full dataset ({total_points} points)...")
        graph_builder = graph_class(distance_matrix, neighbor_index)
        
        print(f"  Computing edges for {sample_size} sampled points...")
        start_time = time.time()
        stats = graph_builder.analyze_sparsity(sampled_point_indices)
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
    print(f"Note: Each of the {sample_size} sampled points can connect to any")
    print(f"of the {total_points} points in the full dataset.")
    print("=" * 80)
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
