"""
Precompute cluster IDs for training data and save as sidecar .pt files.

For each blob_XXXXX.jsonl, creates a corresponding blob_XXXXX.pt containing
cluster IDs for all texts in that blob.

Usage:
    python precompute_clusters.py --data_path data/slimpajama_train --num_files 10
    python precompute_clusters.py --data_path data/slimpajama_train --stride 8  # full dataset
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
import tiktoken

from cluster_assigner import ClusterAssigner


def get_cache_dir(data_path: str, num_clusters: int) -> str:
    """Get the cache directory for a given data path and cluster count."""
    # data/slimpajama_train -> data/slimpajama_train_clusters_1m
    suffix = f"_clusters_{num_clusters // 1000}k" if num_clusters < 1_000_000 else f"_clusters_{num_clusters // 1_000_000}m"
    return data_path + suffix


def precompute_blob(
    blob_path: str,
    assigner: ClusterAssigner,
    tokenizer: tiktoken.Encoding,
    output_path: str,
    stride: int = 8,
    append_eot: bool = True,
) -> dict:
    """
    Precompute cluster IDs for all texts in a blob.
    
    Returns:
        Dict with stats: num_texts, total_tokens, time_s
    """
    # Load all texts from blob
    texts = []
    with open(blob_path, "r") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
    
    # Tokenize all texts
    all_token_ids = []
    for text in texts:
        tokens = tokenizer.encode(text, disallowed_special=())
        if append_eot:
            tokens.append(tokenizer.eot_token)
        all_token_ids.append(tokens)
    
    # Assign clusters
    start = time.time()
    cluster_ids_list = []
    for token_ids in all_token_ids:
        cluster_ids = assigner.assign_single(token_ids, stride=stride)
        cluster_ids_list.append(cluster_ids.cpu())
    elapsed = time.time() - start
    
    # Save as .pt file
    # Format: list of (token_ids, cluster_ids, stride) tuples
    # This allows reconstructing full cluster_ids at any position
    save_data = {
        "cluster_ids": cluster_ids_list,  # List of tensors
        "token_lengths": [len(t) for t in all_token_ids],  # For verification
        "stride": stride,
        "num_centroids": assigner.num_centroids,
        "window_size": assigner.window_size,
    }
    torch.save(save_data, output_path)
    
    total_tokens = sum(len(t) for t in all_token_ids)
    return {
        "num_texts": len(texts),
        "total_tokens": total_tokens,
        "time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute cluster IDs for training data")
    parser.add_argument("--data_path", type=str, default="data/slimpajama_train",
                        help="Path to data directory with blob_XXXXX.jsonl files")
    parser.add_argument("--centroid_path", type=str, 
                        default="experiments/chunk_embeddings/centroids_1m_w32_slimpajama.pt",
                        help="Path to centroid file")
    parser.add_argument("--num_files", type=int, default=None,
                        help="Number of files to process (None = all)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride for cluster assignment (8 = assign every 8th position)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already processed files")
    args = parser.parse_args()
    
    # Load assigner
    print(f"Loading ClusterAssigner from {args.centroid_path}...")
    assigner = ClusterAssigner(args.centroid_path)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Setup output directory
    cache_dir = get_cache_dir(args.data_path, assigner.num_centroids)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Output directory: {cache_dir}")
    
    # Get blob files
    blob_files = sorted(glob.glob(f"{args.data_path}/blob_*.jsonl"))
    if args.num_files:
        blob_files = blob_files[:args.num_files]
    print(f"Processing {len(blob_files)} blob files...")
    
    # Process each blob
    total_tokens = 0
    total_time = 0
    for i, blob_path in enumerate(blob_files):
        blob_name = Path(blob_path).stem  # blob_00000
        output_path = os.path.join(cache_dir, f"{blob_name}.pt")
        
        if args.resume and os.path.exists(output_path):
            print(f"[{i+1}/{len(blob_files)}] Skipping {blob_name} (already exists)")
            continue
        
        print(f"[{i+1}/{len(blob_files)}] Processing {blob_name}...", end=" ", flush=True)
        stats = precompute_blob(blob_path, assigner, tokenizer, output_path, stride=args.stride)
        print(f"{stats['num_texts']} texts, {stats['total_tokens']:,} tokens in {stats['time_s']:.1f}s")
        
        total_tokens += stats["total_tokens"]
        total_time += stats["time_s"]
    
    print(f"\nDone! Processed {total_tokens:,} tokens in {total_time:.1f}s")
    print(f"Cache saved to: {cache_dir}")


if __name__ == "__main__":
    main()

