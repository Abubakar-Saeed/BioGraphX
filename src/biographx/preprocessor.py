"""
SequencePreprocessor
================================================================================

Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Adaptive sequence processing strategy for protein sequences of variable
    lengths. Implements intelligent truncation and sliding window protocols to
    standardize input for downstream analysis while preserving biologically
    critical regions and functional motifs.

    This preprocessor addresses the computational challenge of analyzing
    proteins with extreme length variation (from <100 to >10,000 residues) by
    implementing three tiered strategies:

    1. Direct processing for sequences ≤2000 residues
    2. Smart truncation (40-40-20 distribution) for sequences ≤10000 residues
    3. Sliding window decomposition for sequences >10000 residues

    Critical motif preservation is prioritized throughout all strategies, with
    explicit recovery mechanisms for nuclear localization signals (NLS),
    transmembrane domains, signal peptides, retention signals, and post-
    translational modification sites.

Notes on Distance Metrics:
    IMPORTANT: All motif gap lengths and window positions refer to LINEAR
    SEQUENCE POSITIONS (primary structure indices), not 3D spatial distances.
    This is particularly relevant when evaluating cysteine spacing in disulfide
    patterns (C.{2,10}C) and bipartite signal gap constraints, which represent
    covalent connectivity in primary sequence rather than tertiary proximity.

================================================================================
"""

import re
import numpy as np
from math import log2
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field


class SequencePreprocessor:
    """
    Adaptive sequence processing engine for variable-length protein sequences.

    This class implements intelligent truncation and windowing strategies that
    preserve functionally critical regions while achieving standardized length
    constraints for computational analysis. The processor prioritizes retention
    of known targeting signals, transmembrane domains, and post-translational
    modification motifs.

    Attributes
    ----------
    motif_patterns : Dict[str, List[str]]
        Comprehensive collection of regular expression patterns for biologically
        significant sequence motifs, organized by functional category.
        Categories include:
        - NLS: Nuclear localization signals (monopartite, bipartite)
        - signal_peptide: Secretory pathway targeting sequences
        - tm_domains: Transmembrane helix signatures
        - phosphorylation: Kinase recognition motifs
        - disulfide: Disulfide-bonded cysteine spacing patterns
        - retention_signals: ER/Golgi/lysosomal retention motifs

    critical_lengths : Dict[str, int]
        Defined sequence windows (in residues) for N-terminal, C-terminal, and
        internal regions where functional motifs are most frequently localized.
        These guide preservation priorities during truncation.

    Notes
    -----
    All distance parameters in motif patterns (e.g., C.{2,10}C) refer to LINEAR
    SEQUENCE SEPARATION, not Euclidean distance in folded structures.
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # FUNCTIONAL MOTIF PATTERNS
        # ---------------------------------------------------------------------
        # Regular expression patterns for biologically significant sequence motifs.
        # Patterns are case-insensitive and prioritized for preservation during
        # truncation operations. Each category contains multiple variant patterns
        # to capture canonical and non-canonical signal sequences.
        #
        # IMPORTANT: All gap lengths (e.g., .{2,10}) represent LINEAR SEQUENCE
        # distances, NOT 3D structural measurements.
        # ---------------------------------------------------------------------
        self.motif_patterns = {
            'NLS': [
                r'[KR]\.[KR]',  # Basic residue pair with spacer
                r'P\.[KR]{3,}',  # Proline followed by basic cluster
                r'[KR]{4,}'  # Polybasic cluster (monopartite NLS)
            ],
            'signal_peptide': [
                r'^M[AVILMFYW]{10,}',  # Met + extended hydrophobic region
                r'^M.{1,30}[AVILMFYW]{8,}'  # Flexible signal peptide pattern
            ],
            'tm_domains': [
                r'[AVILMFYW]{15,}',  # Extended hydrophobic stretch (TMD)
                r'[AVILMFYW]{10,}[^AVILMFYW]{1,5}[AVILMFYW]{10,}'  # Split TMD
            ],
            'phosphorylation': [
                r'[ST]P',  # Proline-directed kinase site
                r'[ST]\.P'  # Flexible phosphorylation motif
            ],
            'disulfide': [
                r'C.{2,10}C',  # Cysteine spacing (linear sequence)
                r'C.{4,8}C'  # Typical disulfide-bonded pattern
            ],
            'retention_signals': [
                r'KDEL$',  # ER retention (soluble proteins)
                r'KKXX$',  # Dilysine ER retrieval (membrane)
                r'KK.{2}$',  # Alternative dilysine pattern
                r'Y[^Y]{2}[FILMV]$',  # YXXΦ endocytosis/sorting motif
                r'[DE]XXXL[LI]'  # Dileucine sorting signal
            ]
        }

        # ---------------------------------------------------------------------
        # CRITICAL REGION LENGTHS
        # ---------------------------------------------------------------------
        # Empirically defined sequence windows where functional signals are
        # most frequently concentrated. These regions receive preservation
        # priority during smart truncation operations.
        # ---------------------------------------------------------------------
        self.critical_lengths = {
            'n_terminal': 50,  # Signal peptides, mitochondrial targeting, etc.
            'c_terminal': 30,  # ER retention, PTS1, dilysine motifs
            'internal_motifs': 100  # Disulfide patterns, phosphorylation clusters
        }

    # -------------------------------------------------------------------------
    # SMART TRUNCATION ENGINE
    # -------------------------------------------------------------------------
    def smart_truncate(self, seq: str, target: int = 2000) -> str:
        """
        Perform biologically-aware sequence truncation with motif preservation.

        Implements a 40-40-20 segmentation strategy that preserves:
        - 40% N-terminal region (targeting signals, localization peptides)
        - 40% C-terminal region (retention motifs, organelle targeting)
        - 20% central region (internal structural elements)

        Parameters
        ----------
        seq : str
            Original amino acid sequence.
        target : int, default=2000
            Desired maximum length after truncation.

        Returns
        -------
        str
            Truncated sequence with preserved critical motifs. Length guaranteed
            ≤ target residues.

        Notes
        -----
        If critical motifs present in original are absent from truncated version,
        the method attempts intelligent insertion of up to two representative
        motifs while preserving approximate positional context.
        """
        if len(seq) <= target:
            return seq

        # Calculate segment lengths according to 40-40-20 distribution
        n_len = int(target * 0.4)  # N-terminal preservation (signals, targeting)
        c_len = int(target * 0.4)  # C-terminal preservation (retention, sorting)
        m_len = target - n_len - c_len  # Central region (structural context)

        # Extract prioritized segments
        n_segment = seq[:n_len]  # Preserve start
        c_segment = seq[-c_len:] if c_len > 0 else ""  # Preserve end
        middle_start = (len(seq) - m_len) // 2  # Central region
        m_segment = seq[middle_start:middle_start + m_len]

        # Assemble truncated sequence
        truncated = n_segment + m_segment + c_segment

        # Verify motif preservation
        motifs_found = self.scan_motifs(truncated)

        # Recover critical motifs absent from truncated version
        for motif_type, patterns in motifs_found.items():
            if not patterns:  # No motifs of this type preserved
                # Search original sequence for this motif class
                original_motifs = self._find_motifs_in_sequence(seq, motif_type)
                if original_motifs:
                    # Attempt insertion of up to 2 representative motifs
                    for motif in original_motifs[:2]:
                        if motif not in truncated:
                            truncated = self._insert_motif_smart(truncated, motif)

        return truncated[:target]  # Enforce exact length constraint

    def _insert_motif_smart(self, seq: str, motif: str) -> str:
        """
        Intelligently insert missing motif while preserving sequence context.

        Parameters
        ----------
        seq : str
            Current truncated sequence.
        motif : str
            Motif substring to be inserted.

        Returns
        -------
        str
            Modified sequence with motif inserted at optimal position.

        Notes
        -----
        Attempts insertion at three candidate positions based on typical
        motif localization patterns: near N-terminus (position 30), central
        region (midpoint), and near C-terminus (position -30). Avoids
        duplicate insertions.
        """
        if len(seq) < 50:
            return seq  # Insufficient length for meaningful insertion

        # Candidate insertion positions based on biological context
        insertion_points = [
            min(30, len(seq)),  # Proximal to N-terminus
            len(seq) // 2,  # Central region
            max(len(seq) - 30, 0)  # Proximal to C-terminus
        ]

        for pos in insertion_points:
            test_seq = seq[:pos] + motif + seq[pos:]
            # Verify insertion successful without duplication
            if motif in test_seq and test_seq.count(motif) == 1:
                return test_seq

        return seq  # Return unmodified if insertion fails

    # -------------------------------------------------------------------------
    # SLIDING WINDOW GENERATOR
    # -------------------------------------------------------------------------
    def create_sliding_windows(self, seq: str, window_size: int = 1000,
                               stride: int = 500) -> List[Tuple[str, Dict]]:
        """
        Generate overlapping windows for ultra-long sequence decomposition.

        Parameters
        ----------
        seq : str
            Full-length amino acid sequence.
        window_size : int, default=1000
            Size of each sliding window (residues).
        stride : int, default=500
            Step size between consecutive windows (residues).

        Returns
        -------
        List[Tuple[str, Dict]]
            List of (window_sequence, metadata) pairs where metadata includes:
            - position: (start, end) indices in original sequence
            - motif_score: Information content score
            - contains_critical: Boolean flags for critical region inclusion
            - window_id: Sequential window identifier

        Notes
        -----
        Automatically includes both regularly spaced windows and a final window
        that extends to the C-terminus when sequence length modulo stride ≠ 0.
        """
        if len(seq) <= window_size:
            return [(seq, {'position': (0, len(seq)), 'motif_score': 1.0})]

        windows = []
        n_windows = max(1, (len(seq) - window_size) // stride + 1)

        for i in range(n_windows):
            start = i * stride
            end = min(start + window_size, len(seq))
            window_seq = seq[start:end]

            # Calculate window quality metrics
            motif_score = self._calculate_window_information_content(window_seq)
            contains_critical = self._window_contains_critical(
                window_seq, start, end, len(seq)
            )

            windows.append((
                window_seq,
                {
                    'position': (start, end),
                    'motif_score': motif_score,
                    'contains_critical': contains_critical,
                    'window_id': i
                }
            ))

        # Add final window to ensure C-terminal coverage
        if len(seq) % stride != 0 and len(seq) > window_size:
            final_start = len(seq) - window_size
            final_window = seq[final_start:]
            motif_score = self._calculate_window_information_content(final_window)

            windows.append((
                final_window,
                {
                    'position': (final_start, len(seq)),
                    'motif_score': motif_score,
                    'contains_critical': self._window_contains_critical(
                        final_window, final_start, len(seq), len(seq)
                    ),
                    'window_id': len(windows)
                }
            ))

        return windows

    def _calculate_window_information_content(self, window: str) -> float:
        """
        Quantify information density of a sequence window.

        Combines multiple metrics to assess the biological significance of a
        sequence segment:
        1. Motif density (normalized count of functional patterns)
        2. Sequence complexity (Shannon entropy, normalized to 20aa alphabet)
        3. Hydrophobic cluster conservation (proxy for structural elements)
        4. Charge cluster density (proxy for functional interfaces)

        Parameters
        ----------
        window : str
            Amino acid sequence window.

        Returns
        -------
        float
            Normalized information content score (0.0-1.0). Higher values
            indicate greater biological information density.

        """
        if len(window) == 0:
            return 0.0

        scores = []

        # 1. Motif density score (normalized to 0-1)
        motif_count = 0
        for motif_type, patterns in self.motif_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, window, re.IGNORECASE)
                    motif_count += sum(1 for _ in matches)
                except:
                    continue

        motif_density = motif_count / len(window)
        scores.append(min(motif_density * 10, 1.0))  # 10% motif density → 1.0

        # 2. Shannon entropy (sequence complexity)
        if len(window) > 0:
            freq = Counter(window)
            entropy = -sum((count / len(window)) * log2(count / len(window))
                           for count in freq.values())
            # Normalize to maximum entropy for 20-letter alphabet (log2(20) ≈ 4.32)
            scores.append(min(entropy / log2(20), 1.0))

        # 3. Hydrophobic cluster score (structural element proxy)
        hydrophobic = sum(1 for aa in window if aa in 'AVILMFYW')
        conservation = hydrophobic / len(window)
        scores.append(conservation)

        # 4. Charge cluster density (functional interface proxy)
        charged = sum(1 for aa in window if aa in 'KRHDE')
        charge_score = charged / len(window) if charged > 3 else 0
        scores.append(min(charge_score * 2, 1.0))

        return np.mean(scores) if scores else 0.5

    def _window_contains_critical(self, window: str, start: int, end: int,
                                  total_len: int) -> Dict[str, bool]:
        """
        Assess whether window encompasses biologically critical regions.

        Parameters
        ----------
        window : str
            Sequence window being evaluated.
        start : int
            Start index in original sequence.
        end : int
            End index in original sequence.
        total_len : int
            Total length of original sequence.

        Returns
        -------
        Dict[str, bool]
            Flags indicating inclusion of:
            - n_terminal: Within first 50 residues
            - c_terminal: Within last 30 residues
            - has_motifs: Contains functional motifs
            - middle_region: Central 20% of sequence
        """
        return {
            'n_terminal': start < self.critical_lengths['n_terminal'],
            'c_terminal': end > total_len - self.critical_lengths['c_terminal'],
            'has_motifs': len(self._find_motifs_in_window(window)) > 0,
            'middle_region': (start > total_len * 0.4 and
                              end < total_len * 0.6)
        }

    # -------------------------------------------------------------------------
    # MOTIF DETECTION UTILITIES
    # -------------------------------------------------------------------------
    def _find_motifs_in_window(self, window: str) -> List[str]:
        """
        Identify all functional motifs present in a sequence window.

        Parameters
        ----------
        window : str
            Amino acid sequence window.

        Returns
        -------
        List[str]
            Unique motif sequences found in the window.
        """
        motifs = []
        for motif_type, patterns in self.motif_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, window, re.IGNORECASE)
                    motifs.extend(matches)
                except:
                    continue
        return list(set(motifs))  # Remove duplicates

    def _find_motifs_in_sequence(self, seq: str, motif_type: str) -> List[str]:
        """
        Retrieve all instances of a specific motif class from sequence.

        Parameters
        ----------
        seq : str
            Amino acid sequence.
        motif_type : str
            Category of motif to search for (key in self.motif_patterns).

        Returns
        -------
        List[str]
            Unique motif sequences of the specified type.
        """
        motifs = []
        if motif_type in self.motif_patterns:
            for pattern in self.motif_patterns[motif_type]:
                try:
                    matches = re.findall(pattern, seq, re.IGNORECASE)
                    motifs.extend(matches)
                except:
                    continue
        return list(set(motifs))

    def scan_motifs(self, seq: str) -> Dict[str, List[str]]:
        """
        Comprehensive motif scanning across all functional categories.

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping motif categories to lists of detected motif
            sequences (duplicates removed per category).

        Notes
        -----
        This method performs exhaustive pattern matching across all predefined
        motif categories and returns complete motif inventories. Suitable for
        initial sequence characterization and quality assessment.
        """
        results = {}
        for motif_type, patterns in self.motif_patterns.items():
            found = []
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, seq, re.IGNORECASE)
                    found.extend(matches)
                except:
                    continue
            results[motif_type] = list(set(found))
        return results

    # -------------------------------------------------------------------------
    # ADAPTIVE PROCESSING ORCHESTRATOR
    # -------------------------------------------------------------------------
    def adaptive_process(self, seq: str) -> List[Tuple[str, Dict]]:
        """
        Main orchestrator for adaptive sequence processing.

        Selects and executes the optimal processing strategy based on sequence
        length:

        1. ≤2000 residues: Direct processing (full sequence)
           - Preserves complete sequence context
           - Minimal information loss
           - Returns single sequence with 'full' strategy metadata

        2. 2001-10000 residues: Smart truncation
           - 40-40-20 segmentation with motif preservation
           - Returns single truncated sequence with preservation metadata
           - Recovery of critical motifs if truncated

        3. >10000 residues: Sliding window decomposition
           - Overlapping windows (1000bp window, 500bp stride)
           - Returns multiple windowed sequences with position tracking
           - Enables distributed processing of ultra-long proteins

        Parameters
        ----------
        seq : str
            Input amino acid sequence of any length.

        Returns
        -------
        List[Tuple[str, Dict]]
            Processed sequence(s) with comprehensive metadata including:
            - Processing strategy employed
            - Original/processed length metrics
            - Motif inventory
            - Preservation statistics
            - Window coordinates (for sliding window strategy)

        Notes
        -----
        This method serves as the primary API entry point for sequence
        preprocessing operations. All downstream analyses should route
        sequences through this orchestrator to ensure consistent handling
        of length heterogeneity.
        """
        length = len(seq)

        # -----------------------------------------------------------------
        # STRATEGY 1: DIRECT PROCESSING (≤2000 residues)
        # -----------------------------------------------------------------
        if length <= 2000:
            return [(seq, {
                'strategy': 'full',
                'original_length': length,
                'processed_length': length,
                'motifs': self.scan_motifs(seq)
            })]

        # -----------------------------------------------------------------
        # STRATEGY 2: SMART TRUNCATION (2001-10000 residues)
        # -----------------------------------------------------------------
        elif length <= 10000:
            truncated = self.smart_truncate(seq, 2000)
            return [(truncated, {
                'strategy': 'smart_truncate',
                'original_length': length,
                'processed_length': len(truncated),
                'truncation_ratio': len(truncated) / length,
                'motifs': self.scan_motifs(truncated),
                'preserved_regions': {
                    'n_terminal': True,
                    'c_terminal': True,
                    'middle': True
                }
            })]

        # -----------------------------------------------------------------
        # STRATEGY 3: SLIDING WINDOW (>10000 residues)
        # -----------------------------------------------------------------
        else:
            windows = self.create_sliding_windows(seq, 1000, 500)
            return [(window, {
                'strategy': 'sliding_window',
                'window_info': info,
                'original_length': length,
                'total_windows': len(windows),
                'window_index': info['window_id'],
                'motifs': self.scan_motifs(window)
            }) for window, info in windows]
