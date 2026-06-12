"""
BioGraphXPipeline
================================================================================

Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Master orchestration pipeline integrating all components of the BioGraphX
    feature extraction system. This conductor class coordinates the sequential
    and parallel execution of:

    1. Sequence preprocessing (adaptive length normalization)
    2. Biophysical property calculation (hydrophobicity, charge, pI, etc.)
    3. Residue interaction graph construction (12 interaction types)
    4. Hybrid interaction scoring (synergistic motif combinations)
    5. Subcellular localization motif profiling
    6. Configurational frustration analysis

    The pipeline implements three adaptive processing strategies based on
    sequence length, enabling scalable feature extraction from proteins ranging
    from <100 to >10,000 residues. All extracted features are concatenated into
    a fixed-length vector (147+11 dimensions) compatible with downstream machine
    learning classifiers.

Notes on Distance Metrics:
    IMPORTANT: Throughout this pipeline, all graph edge distance constraints
    refer to LINEAR SEQUENCE SEPARATION (|pos_i - pos_j|) in the primary
    structure, NOT 3D Euclidean distances (angstroms). This is a critical
    distinction when interpreting interaction networks and frustration patterns.

================================================================================
"""
from typing import Dict,List

from biographx.biophysics import BioPhysicsStrategy
from biographx.preprocessor import SequencePreprocessor
from biographx.profiler import MotifProfiler
from biographx.graph_engine import GraphEngine
from biographx.frustration_analyzer import FrustrationAnalyzer
from biographx.utils.feature_names import COMPLETE_FEATURE_NAMES
import numpy as np
import gc
import csv
import pandas as pd
from joblib import Parallel, delayed

class BioGraphXPipeline:
    """
    Master orchestration pipeline for integrated protein feature extraction.

    This conductor class coordinates the complete feature extraction workflow,
    binding together biophysical computation, graph-theoretic analysis, motif
    detection, and frustration analysis into a unified processing stream.

    Attributes
    ----------
    biophysics : BioPhysicsStrategy
        Provides physicochemical property scales and calculation methods.
    preprocessor : SequencePreprocessor
        Handles adaptive length normalization and motif preservation.
    motif_profiler : MotifProfiler
        Detects subcellular localization signals and targeting motifs.
    graph_engine : GraphEngine
        Constructs residue interaction graphs with 12 edge types.
    frustration_analyzer : FrustrationAnalyzer
        Computes configurational frustration metrics from constraint graphs.

    total_features : int
        Total dimensionality of concatenated feature vector (157).
        Derived from:
        - Basic graph features: 85 dimensions
        - Hybrid interaction features: 22 dimensions
        - Targeting signal features: 20 dimensions (2 scores × 10 compartments)
        - Biophysical features: 19 dimensions
        - Frustration features: 11 dimensions

    Notes
    -----
    The pipeline automatically handles sequences of arbitrary length through
    three adaptive strategies. All distance thresholds within graph construction
    are defined as LINEAR SEQUENCE DISTANCES, not 3D structural measurements.
    """

    def __init__(self):
        """
        Initialize the complete BioGraphX pipeline with all component modules.

        Creates and connects instances of each analytical component through
        dependency injection. The graph engine and frustration analyzer receive
        the biophysics strategy instance for consistent physicochemical parameters.
        """
        # -----------------------------------------------------------------
        # Component Initialization
        # -----------------------------------------------------------------
        self.biophysics = BioPhysicsStrategy()
        self.preprocessor = SequencePreprocessor()
        self.motif_profiler = MotifProfiler(self.biophysics)
        self.graph_engine = GraphEngine(self.biophysics, self.motif_profiler)
        self.frustration_analyzer = FrustrationAnalyzer(self.biophysics)

        # -----------------------------------------------------------------
        # Feature Dimensionality
        # -----------------------------------------------------------------
        # Total concatenated feature vector length (157 dimensions)
        # This fixed-length representation ensures compatibility with
        # downstream classifiers regardless of input sequence length.
        self.total_features = 157                               # 157 total features

    # -------------------------------------------------------------------------
    # CORE FEATURE EXTRACTION ENGINE
    # -------------------------------------------------------------------------
    def extract_full_features(self, sequence: str) -> np.ndarray:
        """
        Extract complete feature vector from a single protein sequence.

        Executes the full feature extraction pipeline without length adaptation:
        1. Build residue interaction graph with hybrid interaction tracking
        2. Compute configurational frustration metrics
        3. Extract topological features from interaction graph
        4. Calculate hybrid interaction enrichment scores
        5. Generate subcellular targeting signal features
        6. Compute global biophysical property vector
        7. Concatenate and validate feature dimensionality

        Parameters
        ----------
        sequence : str
            Amino acid sequence (single-letter codes). Should be preprocessed
            to appropriate length for direct processing.

        Returns
        -------
        np.ndarray (dtype=np.float32)
            Fixed-length feature vector of 157 dimensions. Guaranteed to be
            NaN-free with all values in valid numerical range.

        Notes
        -----
        This method is designed for sequences ≤2000 residues. For longer
        sequences, use adaptive_extract_features() which implements smart
        truncation or sliding window aggregation.
        """
        # -----------------------------------------------------------------
        # Step 1: Graph Construction and Hybrid Scoring
        # -----------------------------------------------------------------
        # Build complete residue interaction graph with all 12 interaction types
        # hybrid_scores contains normalized enrichment values for 9 hybrid classes
        graph, hybrid_scores = self.graph_engine.build_complete_graph(sequence)

        # -----------------------------------------------------------------
        # Step 2: Configurational Frustration Analysis
        # -----------------------------------------------------------------
        # Compute frustration metrics from constraint satisfaction patterns
        frustration_features = self.frustration_analyzer.compute_from_constraint_graph(
            graph, sequence
        )

        # -----------------------------------------------------------------
        # Step 3: Graph Feature Extraction
        # -----------------------------------------------------------------
        # Extract topological features (density, clustering, centrality, etc.)
        basic_features = self.graph_engine.extract_basic_graph_features(graph)

        # -----------------------------------------------------------------
        # Step 4: Hybrid Interaction Features
        # -----------------------------------------------------------------
        # Compute enrichment scores for synergistic interaction combinations
        hybrid_features = self.graph_engine.extract_hybrid_features(
            graph, hybrid_scores, sequence
        )

        # -----------------------------------------------------------------
        # Step 5: Localization Targeting Signal Features
        # -----------------------------------------------------------------
        # Generate motif-based and interaction-based localization evidence
        targeting_signal_features = self.motif_profiler.extract_targeting_signal_features(
            sequence, hybrid_scores
        )

        # -----------------------------------------------------------------
        # Step 6: Global Biophysical Properties
        # -----------------------------------------------------------------
        # Compute pI, GRAVY, charge, autocorrelation, entropy, etc.
        physics = self.biophysics.extract_global_physics(sequence)

        # -----------------------------------------------------------------
        # Step 7: Frustration Vector Conversion
        # -----------------------------------------------------------------
        # Collapse frustration dictionary to fixed-length numerical vector
        frustration_vector = self._extract_frustration_vector(frustration_features)

        # -----------------------------------------------------------------
        # Step 8: Feature Concatenation
        # -----------------------------------------------------------------
        # Combine all feature subsets in deterministic order
        all_features = np.concatenate([
            basic_features,  # 12 dimensions
            hybrid_features,  # 7 dimensions
            targeting_signal_features,  # 20 dimensions
            physics,  # 19 dimensions
            frustration_vector  # 11 dimensions
        ])

        # -----------------------------------------------------------------
        # Step 9: Dimensionality Validation
        # -----------------------------------------------------------------
        # Ensure exact feature count (157 = 146 + 11)
        target_dim = 146 + 11
        if len(all_features) < target_dim:
            # Pad with zeros if insufficient features (should not occur in normal operation)
            print(f"[Warning] Feature padding: {target_dim - len(all_features)} zeros added")
            all_features = np.pad(
                all_features, (0, target_dim - len(all_features)), 'constant'
            )
        elif len(all_features) > target_dim:
            # Truncate if feature vector exceeds expected dimension
            print(f"[Warning] Feature truncation: trimmed from {len(all_features)} to {target_dim}")
            all_features = all_features[:target_dim]

        # -----------------------------------------------------------------
        # Step 10: Numerical Sanitization
        # -----------------------------------------------------------------
        # Replace any NaN, Inf, -Inf with 0.0 to ensure classifier compatibility
        all_features = np.nan_to_num(
            all_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        return all_features.astype(np.float32)

    # -------------------------------------------------------------------------
    # FRUSTRATION FEATURE PARSING
    # -------------------------------------------------------------------------
    def _extract_frustration_vector(self, frustration_features: Dict) -> np.ndarray:
        """
        Convert frustration analysis dictionary to fixed-length numerical vector.

        Parameters
        ----------
        frustration_features : Dict
            Comprehensive frustration metrics from FrustrationAnalyzer,
            containing per-residue vectors and summary statistics.

        Returns
        -------
        np.ndarray
            11-dimensional vector encoding:
            [0]  : N-terminal region mean frustration
            [1]  : C-terminal region mean frustration
            [2]  : Structural core mean frustration
            [3]  : Signal region vs structural frustration difference
            [4]  : Maximum frustration in signal peptide
            [5]  : Localization domain frustration score
            [6]  : Frustration hotspot count
            [7]  : Constraint satisfaction ratio
            [8]  : Frustration profile correlation (if available)
            [9]  : High-frustration signal region score
            [10] : Global mean frustration (per-residue average)

        Notes
        -----
        The per-residue frustration vector is reduced to its mean value for
        dimensionality constraints. For per-position analysis, access the
        raw 'Frustration_PerResidue_Vector' directly.
        """
        core_features = [
            frustration_features['Frustration_NTerminal_Mean'],
            frustration_features['Frustration_CTerminal_Mean'],
            frustration_features['Frustration_Structural_Mean'],
            frustration_features['Frustration_SignalVsStructure'],
            frustration_features['Frustration_MaxInSignalRegion'],
            frustration_features['Frustration_Localization'],
            frustration_features['Frustration_HotspotCount'],
            frustration_features['Frustration_SatisfactionRatio'],
            frustration_features.get('Frustration_ProfileCorrelation', 0.0),
            frustration_features['Frustration_HighSignalFrustration'],
            np.mean(frustration_features['Frustration_PerResidue_Vector'])  # Summary statistic
        ]

        return np.array(core_features, dtype=np.float32)

    # -------------------------------------------------------------------------
    # ADAPTIVE PROCESSING ORCHESTRATOR
    # -------------------------------------------------------------------------
    def adaptive_extract_features(self, sequence: str) -> np.ndarray:
        """
        Adaptive feature extraction with length-based strategy selection.

        Implements three-tiered processing strategy based on sequence length:

        1. ≤2,000 residues: Direct full processing
           - Preserves complete sequence information
           - Maximum fidelity for downstream analysis

        2. 2,001-10,000 residues: Smart truncation
           - 40-40-20 segmentation preserving terminal signals
           - Automatic motif recovery for critical patterns
           - Single pass processing

        3. >10,000 residues: Sliding window with weighted aggregation
           - Overlapping windows (1000bp window, 500bp stride)
           - Parallel window processing
           - Information-content weighted averaging
           - Preserves context for ultra-long proteins

        Parameters
        ----------
        sequence : str
            Input amino acid sequence of arbitrary length.

        Returns
        -------
        np.ndarray
            Fixed-length feature vector (157 dimensions) regardless of input
            sequence length. Window aggregation ensures consistent dimensionality.

        Notes
        -----
        For ultra-long proteins (>10,000 residues), the sliding window approach
        with weighted averaging provides robust feature estimation while
        maintaining computational tractability.
        """
        length = len(sequence)

        # -----------------------------------------------------------------
        # STRATEGY 1: DIRECT PROCESSING (≤2000 residues)
        # -----------------------------------------------------------------
        if length <= 2000:
            return self.extract_full_features(sequence)

        # -----------------------------------------------------------------
        # STRATEGY 2: SMART TRUNCATION (2001-10000 residues)
        # -----------------------------------------------------------------
        elif length <= 10000:
            truncated = self.preprocessor.smart_truncate(sequence, 2000)
            return self.extract_full_features(truncated)

        # -----------------------------------------------------------------
        # STRATEGY 3: SLIDING WINDOW (>10000 residues)
        # -----------------------------------------------------------------
        else:
            # Generate overlapping windows with position metadata
            windows_info = self.preprocessor.create_sliding_windows(sequence, 1000, 500)
            window_features = []
            window_weights = []

            # Parallel window processing for computational efficiency
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for window, info in windows_info:
                    future = executor.submit(self.extract_full_features, window)
                    futures.append((future, info))

                # Collect results and extract information weights
                for future, info in futures:
                    features = future.result()
                    window_features.append(features)
                    # Weight by motif density/information content
                    window_weights.append(info['motif_score'])

            # Convert to numpy arrays for vectorized operations
            window_features = np.array(window_features)
            window_weights = np.array(window_weights)

            # Weighted aggregation prioritizing information-rich windows
            if window_weights.sum() > 0:
                window_weights = window_weights / window_weights.sum()
                aggregated = np.average(window_features, axis=0, weights=window_weights)
            else:
                # Fallback to simple average if no motif information
                aggregated = np.mean(window_features, axis=0)

            return aggregated

    # -------------------------------------------------------------------------
    # SINGLE SEQUENCE PROCESSING INTERFACE
    # -------------------------------------------------------------------------
    def process_sequence(self, seq: str) -> np.ndarray:
        """
        Public interface for single sequence feature extraction.

        Parameters
        ----------
        seq : str
            Amino acid sequence (single-letter codes).

        Returns
        -------
        np.ndarray
            Complete feature vector (157 dimensions) for the input sequence.

        Notes
        -----
        This method automatically applies the appropriate adaptive strategy
        based on sequence length. Use for individual protein analysis.
        """
        return self.adaptive_extract_features(seq)

    # -------------------------------------------------------------------------
    # BATCH PROCESSING INTERFACE
    # -------------------------------------------------------------------------
    def process_batch(self, sequences: List[str]) -> List[np.ndarray]:
        """
        Process multiple sequences sequentially within a single batch.

        Parameters
        ----------
        sequences : List[str]
            List of amino acid sequences to process.

        Returns
        -------
        List[np.ndarray]
            List of feature vectors corresponding to each input sequence.

        Notes
        -----
        This method is designed for use with parallel processing frameworks
        (joblib, multiprocessing) where each worker processes a batch
        sequentially. For true parallel processing across sequences, combine
        with Parallel() from joblib.
        """
        results = []
        for seq in sequences:
            features = self.process_sequence(seq)
            results.append(features)
        return results


# =============================================================================
# INTEGRATED BATCH PROCESSING PIPELINE
# =============================================================================

def run_integrated_pipeline(input_file: str, output_file: str,
                            chunk_size: int = 500, n_jobs: int = 4) -> None:
    """
    Execute complete BioGraphX feature extraction pipeline on large CSV datasets.

    This high-throughput function processes protein sequences from a CSV file,
    extracts 157-dimensional feature vectors, and appends them to the original
    data while preserving all non-sequence columns. Implements chunked processing
    for memory efficiency and parallel execution for computational performance.

    Parameters
    ----------
    input_file : str
        Path to input CSV file. Must contain a column named 'Sequence' with
        amino acid sequences. All other columns are preserved without modification.
    output_file : str
        Path to output CSV file. Will contain all original columns (excluding
        Sequence) plus 157 feature columns named according to COMPLETE_FEATURE_NAMES.
    chunk_size : int, default=500
        Number of sequences to process per chunk. Lower values reduce memory
        usage at cost of I/O overhead.
    n_jobs : int, default=4
        Number of parallel worker processes for batch processing. Adjust based
        on available CPU cores.

    Returns
    -------
    None
        Writes results directly to output_file.

    Notes
    -----
    CRITICAL: The 'Sequence' column is USED FOR FEATURE EXTRACTION but DROPPED
    FROM OUTPUT. Ensure your input file contains this column and that you do
    NOT require sequence retention in output.

    Memory management: Intermediate chunks are explicitly garbage-collected to
    prevent memory accumulation during long runs on large datasets.
    """

    # -----------------------------------------------------------------
    # Initialization and Validation
    # -----------------------------------------------------------------
    # Count total rows for progress tracking (exclude header)
    with open(input_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1

    print("=" * 70)
    print("BIOGRAPHX INTEGRATED FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Feature dimensionality: {len(COMPLETE_FEATURE_NAMES)}")
    print(f"Total sequences: {total_rows:,}")
    print(f"Chunk size: {chunk_size}")
    print(f"Parallel workers: {n_jobs}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("=" * 70)

    # Initialize pipeline instance
    pipeline = BioGraphXPipeline()

    # -----------------------------------------------------------------
    # Header Processing
    # -----------------------------------------------------------------
    # Read only header to preserve original column structure
    header_df = pd.read_csv(input_file, nrows=0)

    # ❌ Explicitly EXCLUDE 'Sequence' column from output
    # The sequence is used for feature extraction but not retained in results
    output_columns = [c for c in header_df.columns if c != "Sequence"]

    # Write output header with preserved original columns + feature names
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_columns + COMPLETE_FEATURE_NAMES)

    total_processed = 0

    # -----------------------------------------------------------------
    # Main Processing Loop (Chunked)
    # -----------------------------------------------------------------
    for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):

        # Extract sequences for feature encoding
        sequences = chunk["Sequence"].tolist()

        # Dynamic batch sizing for load balancing
        batch_size = max(50, len(sequences) // (n_jobs * 2))
        batches = [sequences[i:i + batch_size]
                   for i in range(0, len(sequences), batch_size)]

        # Parallel batch processing
        encoded_batches = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            verbose=0
        )(delayed(pipeline.process_batch)(batch) for batch in batches)

        # Flatten batch results
        encoded_vectors = [vec for batch in encoded_batches for vec in batch]

        # ❌ Drop Sequence column before writing output
        chunk_no_seq = chunk.drop(columns=["Sequence"])

        # Append results to output file
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            for i, vector in enumerate(encoded_vectors):
                writer.writerow(chunk_no_seq.iloc[i].tolist() + vector.tolist())

        # Update progress tracking
        total_processed += len(sequences)
        progress_pct = (total_processed / total_rows) * 100
        print(
            f"\rProgress: {progress_pct:.1f}% "
            f"| {total_processed:,}/{total_rows:,} "
            f"| Chunk {chunk_idx + 1}",
            end=""
        )

        # -----------------------------------------------------------------
        # Memory Management
        # -----------------------------------------------------------------
        # Explicitly delete large objects and force garbage collection
        del chunk, chunk_no_seq, batches, encoded_batches, encoded_vectors
        gc.collect()

    print("\n\nPipeline execution complete!")
    print(f"Output saved to: {output_file}")

    # -----------------------------------------------------------------
    # Output Validation
    # -----------------------------------------------------------------
    # Perform sanity check on output file
    df_check = pd.read_csv(output_file, nrows=3)
    print("\nOutput validation:")
    print(f"   - Sequence column in output? {'Sequence' in df_check.columns} (should be False)")
    print(f"   - Total columns: {len(df_check.columns)}")
    print(f"   - Feature columns: {len(COMPLETE_FEATURE_NAMES)}")
    print(f"   - Preserved columns: {len(output_columns)}")
    print("=" * 70)