"""
Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Entry point script for BioGraphX protein sequence analysis pipeline.
    Executes end-to-end feature extraction from protein sequences for subcellular
    localization prediction.

    This script initializes the processing environment, configures input/output paths,
    and invokes the integrated pipeline with parallel processing support.
"""

import sys
import os
import argparse

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Add current script directory to Python path for module discovery
# This enables importing from the biographx package regardless of execution context
sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

from biographx.pipeline import run_integrated_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the BioGraphX integrated feature extraction pipeline."
    )
    default_input = os.path.join(os.path.dirname(__file__), "biographx", "data", "hpa_testset.csv")
    default_output = os.path.join(os.path.dirname(__file__), "biographx", "processed_data", "hpa_test_encoded.csv")

    parser.add_argument(
        "--input-file",
        default=default_input,
        help="Path to the input CSV file containing protein sequences.",
    )
    parser.add_argument(
        "--output-file",
        default=default_output,
        help="Path to write the encoded output feature matrix CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of sequences to process per chunk.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel worker processes to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_integrated_pipeline(
        input_file=args.input_file,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
