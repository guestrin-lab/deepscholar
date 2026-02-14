#!/usr/bin/env python3
"""
Generate queries.csv from papers_with_related_works.csv

This script generates queries for DeepScholar-Base from a dataset of papers.
Each query asks to write a related works section based on the paper's abstract.

Usage:
    python generate_queries.py --input dataset/papers_with_related_works.csv --output dataset/queries.csv
"""

import argparse
import pandas as pd
from pathlib import Path


QUERY_TEMPLATE = """Your task is to write a Related Works section for an academic paper given the paper's abstract. Your response should provide the Related Works section and references. Only include references from arXiv that are published before {cutoff_date}. Mention them in a separate, numbered reference list at the end and use the reference numbers to provide in-line citations in the Related Works section for all claims referring to a source (e.g., description of source [3]. Further details [6][7][8][9][10].) Each in-line citation must consist of a single reference number within a pair of brackets. Do not use any other citation format. Do not exceed 600 words for the related works section. Here is the paper abstract:
{abstract}"""


def generate_queries(input_path: str, output_path: str) -> None:
    """Generate queries.csv from papers_with_related_works.csv"""
    print(f"Loading papers from {input_path}...")
    papers_df = pd.read_csv(input_path)

    print(f"Loaded {len(papers_df)} papers")

    # Generate queries
    print("Generating queries...")
    queries_df = papers_df.copy()

    queries_df["query"] = queries_df.apply(
        lambda row: QUERY_TEMPLATE.format(
            cutoff_date=row["published_date"],
            abstract=row["abstract"]
        ),
        axis=1
    )

    # Save queries
    queries_df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(queries_df)} queries")
    print(f"✅ Saved to {output_path}")

    # Print sample
    print("\nSample query (first 500 chars):")
    print("=" * 60)
    print(queries_df["query"].iloc[0][:500] + "...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate queries.csv from papers_with_related_works.csv"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file (papers_with_related_works.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file (queries.csv)",
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate queries
    try:
        generate_queries(str(input_path), str(output_path))
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
