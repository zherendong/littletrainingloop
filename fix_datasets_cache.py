"""
Fix datasets cache by clearing old cached metadata and re-downloading.
"""

import os
import shutil
from datasets import config

def clear_dataset_cache(dataset_name):
    """Clear cache for a specific dataset."""
    cache_dir = config.HF_DATASETS_CACHE
    
    # Try different possible cache directory names
    possible_names = [
        dataset_name,
        dataset_name.replace('/', '___'),
        dataset_name.replace('/', '__'),
    ]
    
    for name in possible_names:
        dataset_cache = os.path.join(cache_dir, name)
        if os.path.exists(dataset_cache):
            print(f"Removing cache: {dataset_cache}")
            shutil.rmtree(dataset_cache)
            print(f"✓ Cleared cache for {dataset_name}")
            return True
    
    print(f"No cache found for {dataset_name}")
    return False


if __name__ == "__main__":
    print("Clearing datasets cache to fix version incompatibility...")
    print()

    # Clear caches for common benchmark datasets
    datasets_to_clear = [
        "Rowan/hellaswag",
        "hellaswag",
        "allenai/arc",
        "arc",
        "cais/mmlu",
        "mmlu",
    ]

    for dataset in datasets_to_clear:
        clear_dataset_cache(dataset)

    # Also clear any remaining cached datasets with old metadata
    cache_dir = config.HF_DATASETS_CACHE
    print(f"\nScanning {cache_dir} for other cached datasets...")
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path) and '___' in item:
                print(f"Found cached dataset: {item}")
                # Check if it has old metadata by looking for dataset_info.json
                info_file = os.path.join(item_path, "dataset_info.json")
                if os.path.exists(info_file):
                    try:
                        import json
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                            # Check if it has the old 'List' feature type
                            if '"_type": "List"' in json.dumps(info):
                                print(f"  Removing {item} (has old 'List' feature type)")
                                shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"  Error checking {item}: {e}")

    print()
    print("Cache cleared! The datasets will be re-downloaded on next use.")
