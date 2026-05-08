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

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Add src directory to Python path for module discovery
# This enables importing from biographx package regardless of execution context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

from biographx.pipeline import run_integrated_pipeline

# Input configuration
input_file = r"C:\Users\abubakar\BioGraphX\BioGraphX-Encoding\src\biographx\data\hpa_testset.csv"
output_file = r"C:\Users\abubakar\BioGraphX\BioGraphX-Encoding\src\biographx\processed_data\hpa_test_encoded.csv"

# Create output directory structure if not exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Execute main processing pipeline
run_integrated_pipeline(
    input_file=input_file,      # Path to input CSV with protein sequences
    output_file=output_file,    # Path for output feature matrix
    chunk_size=500,            # Sequences per processing chunk
    n_jobs=10                  # Parallel worker processes
)
