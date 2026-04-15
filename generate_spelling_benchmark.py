"""
Generate the spelling benchmark dataset.

Usage:
    python generate_spelling_benchmark.py
    python generate_spelling_benchmark.py --samples 2000 --seed 123
"""
import argparse
from spelling_benchmark.generate_data import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic spelling benchmark (count, index, reverse tasks)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="spelling_benchmark/spelling_bee.jsonl",
        help="Output path for the JSONL file"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    generate_dataset(args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
