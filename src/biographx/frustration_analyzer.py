"""
Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Frustration analysis engine for quantifying conformational constraints and
    energetic conflicts in protein sequences. Implements per-residue frustration
    metrics derived from constraint graph topologies.

    This module operationalizes the hypothesis that localized frustration hotspots
    provide the physical basis for overriding sequence-profile incompatibilities
    in moonlighting proteins, particularly those requiring dual localization
    (e.g., Golgi/Plastid sorting).

"""

import igraph as ig
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any

from src.biographx.biophysics import BioPhysicsStrategy


class FrustrationAnalyzer:
    """
    Computes per-residue local frustration metrics from residue interaction constraint graphs.

    This analyzer implements frustration theory adapted for sequence-based analysis,
    treating variance in interaction energies as a proxy for conformational conflict.
    All distance calculations represent linear sequence separation (amino acid positions),
    not spatial proximity in 3D Ångström coordinates.

    The core hypothesis posits that high local frustration in signal regions (N/C-termini)
    provides the necessary energetic override to compensate for poor sequence-profile
    matches in moonlighting proteins targeted to multiple compartments.

    Attributes:
        biophysics (BioPhysicsStrategy): Reference to biophysical rule repository
            containing interaction strengths and physicochemical scales.
    """

    def __init__(self, biophysics_strategy: BioPhysicsStrategy):
        """
        Initialize frustration analyzer with biophysics rule repository.

        Args:
            biophysics_strategy: Centralized biophysical constants and interaction rules
        """
        self.biophysics = biophysics_strategy

    def compute_from_constraint_graph(self, graph: ig.Graph, sequence: str) -> Dict[str, float]:
        """
        Compute comprehensive frustration features from residue interaction constraint graph.

        Transforms graph-encoded residue interaction networks into per-residue
        frustration scores and derived statistical features. Distances in the constraint
        graph represent linear sequence positions, not 3D spatial measurements.

        Args:
            graph: Complete constraint graph from GraphEngine.build_complete_graph()
                  Contains vertices for residues, edges for potential interactions
            sequence: Primary amino acid sequence (single-letter codes)

        Returns:
            Dict[str, float]: Frustration features including:
                - Regional frustration means (N-term, C-term, structural core)
                - Signal-to-structure frustration contrast
                - Frustration localization and hotspot metrics
                - Profile correlation (hypothesis test statistic)
                - Binary signal region frustration flag
                - Fixed-length per-residue frustration vector (100-dim)

        Notes:
            Returns default zero-valued features for sequences with <10 residues
            or empty constraint graphs.
        """
        n = len(sequence)
        if n < 10 or graph.ecount() == 0:
            return self._get_default_frustration_features()

        # Extract edge attributes; default to unit weight/empty type if missing
        edge_weights = graph.es["weight"] if "weight" in graph.es.attributes() else [1.0] * graph.ecount()
        interaction_types = graph.es["interaction_type"] if "interaction_type" in graph.es.attributes() else [
                                                                                                                 ""] * graph.ecount()

        # Core frustration computation
        per_residue_frustration = self._compute_per_residue_frustration(
            graph, edge_weights, interaction_types, sequence
        )

        # Feature extraction pipeline
        features = self._extract_frustration_features(
            per_residue_frustration, sequence, graph, edge_weights
        )

        return features

    def _compute_per_residue_frustration(self, graph: ig.Graph,
                                         edge_weights: List[float],
                                         interaction_types: List[str],
                                         sequence: str) -> np.ndarray:
        """
        Calculate local frustration scores for each residue position.

        Frustration is modeled as the variance in normalized interaction energies
        incident to each residue. High variance indicates conflicting constraints
        that cannot be simultaneously satisfied in a single conformation.

        Args:
            graph: Constraint graph with residue vertices
            edge_weights: Interaction weight values per edge
            interaction_types: Interaction category per edge
            sequence: Primary sequence (unused in computation but preserved for API)

        Returns:
            np.ndarray: Normalized frustration scores in [0,1] range for each residue
                       position. Higher values indicate greater local conflict.

        Notes:
            - Scores normalized to [0,1] by maximum observed frustration
            - Zero frustration for residues with ≤1 incident interaction
            - Interaction strengths modulate edge weights by type-specific coefficients
        """
        n = graph.vcount()
        frustration = np.zeros(n)

        for pos in range(n):
            # Collect all edges incident to current residue
            incident_edges = graph.incident(pos)

            if not incident_edges:
                frustration[pos] = 0
                continue

            # Compute energy proxy: weight * interaction-type strength
            energies = []
            for edge_id in incident_edges:
                weight = edge_weights[edge_id]
                interaction = interaction_types[edge_id]

                # Modulate by interaction-type specific strength coefficient
                strength = self.biophysics.interaction_rules.get(interaction, {}).get('strength', 1.0)
                energies.append(weight * strength)

            # Frustration = variance in interaction energies (conflict metric)
            if len(energies) > 1:
                frustration[pos] = np.var(energies)
            else:
                frustration[pos] = 0

        # Normalize to [0,1] range
        max_frustration = frustration.max() if frustration.max() > 0 else 1
        frustration = frustration / max_frustration

        return frustration

    def _extract_frustration_features(self,
                                      per_residue_frustration: np.ndarray,
                                      sequence: str,
                                      graph: ig.Graph,
                                      edge_weights: List[float]) -> Dict[str, Any]:
        """
        Extract hypothesis-relevant statistical features from per-residue frustration scores.

        Transforms raw frustration vector into interpretable features for
        machine learning classification of moonlighting proteins.

        Args:
            per_residue_frustration: Normalized frustration scores per position
            sequence: Amino acid sequence
            graph: Constraint graph (used for edge count statistics)
            edge_weights: Edge weights (unused but preserved for API)

        Returns:
            Dict[str, Any]: Feature dictionary containing:
                - Regional mean frustrations (float)
                - Frustration contrast metrics (float)
                - Peak frustration in signal regions (float)
                - Frustration localization entropy (float)
                - Hotspot counts (int)
                - Satisfaction ratios (float)
                - Profile correlation statistics (float)
                - Binary classification flags (float)
                - Fixed-length frustration vector (np.ndarray)
        """
        n = len(sequence)
        features = {}

        # 1. Regional frustration analysis
        # Signal regions: N-terminal (first 30 residues) and C-terminal (last 10 residues)
        # Structural core: region between signal regions
        n_term_region = per_residue_frustration[:min(30, n)]
        c_term_region = per_residue_frustration[max(0, n - 10):n]
        middle_region = per_residue_frustration[min(30, n // 2):max(n - 10, 2 * n // 3)]

        features['Frustration_NTerminal_Mean'] = np.mean(n_term_region) if len(n_term_region) > 0 else 0
        features['Frustration_CTerminal_Mean'] = np.mean(c_term_region) if len(c_term_region) > 0 else 0
        features['Frustration_Structural_Mean'] = np.mean(middle_region) if len(middle_region) > 0 else 0

        # 2. Frustration contrast: signal region vs structural core
        # Positive values indicate higher frustration in sorting signals
        features['Frustration_SignalVsStructure'] = (
                features['Frustration_NTerminal_Mean'] - features['Frustration_Structural_Mean']
        )

        # 3. Maximum frustration intensity in signal region
        features['Frustration_MaxInSignalRegion'] = (
            np.max(n_term_region) if len(n_term_region) > 0 else 0
        )

        # 4. Frustration localization (inverse entropy)
        # Low entropy = frustration concentrated at few positions
        frustration_entropy = stats.entropy(per_residue_frustration + 1e-10)
        features['Frustration_Localization'] = 1 / (frustration_entropy + 1e-5)

        # 5. Frustration hotspot identification
        # Hotspots: positions with frustration > 2σ above mean
        frustration_mean = np.mean(per_residue_frustration)
        frustration_std = np.std(per_residue_frustration)
        hotspot_threshold = frustration_mean + 2 * frustration_std
        features['Frustration_HotspotCount'] = int(np.sum(per_residue_frustration > hotspot_threshold))

        # 6. Constraint satisfaction ratio
        # Proportion of residues with frustration below median
        total_interactions = graph.ecount()
        if total_interactions > 0:
            satisfied = np.sum(per_residue_frustration < np.median(per_residue_frustration))
            features['Frustration_SatisfactionRatio'] = satisfied / n
        else:
            features['Frustration_SatisfactionRatio'] = 0

        # 7. Frustration-Profile Conflict Score
        if hasattr(self, '_compute_profile_scores'):
            profile_scores = self._compute_profile_scores(sequence)
            min_len = min(len(per_residue_frustration), len(profile_scores))
            frustration_profile_correlation = np.corrcoef(
                per_residue_frustration[:min_len],
                profile_scores[:min_len]
            )[0, 1] if min_len > 1 else 0
            # Negated so higher = stronger override potential
            features['Frustration_ProfileCorrelation'] = -frustration_profile_correlation

        # 8. Binary flag for high signal region frustration
        # Used as direct input for classification branches
        frustration_threshold = np.percentile(per_residue_frustration, 75) if n > 10 else 0.5
        features['Frustration_HighSignalFrustration'] = float(
            features['Frustration_NTerminal_Mean'] > frustration_threshold
        )

        # 9. Fixed-length per-residue frustration vector
        # Pads or truncates to consistent 100-dim representation
        fixed_length = 100
        if n >= fixed_length:
            features['Frustration_PerResidue_Vector'] = per_residue_frustration[:fixed_length]
        else:
            padded = np.pad(per_residue_frustration, (0, fixed_length - n), 'constant')
            features['Frustration_PerResidue_Vector'] = padded

        return features

    def _compute_profile_scores(self, sequence: str) -> np.ndarray:
        """
        Compute residue-wise sequence profile scores (placeholder implementation).

        Generates normalized hydrophobicity profile as a proxy for sequence
        compatibility scores. In production, this would be replaced by the
        actual motif profiler output.

        Args:
            sequence: Amino acid sequence

        Returns:
            np.ndarray: Normalized profile scores in [0,1] range
                        Higher values indicate stronger motif/pattern match

        Notes:
            This is a minimal implementation for hypothesis testing.
            Full implementation will be done from profiler class
        """
        n = len(sequence)
        scores = np.zeros(n)

        # Simple hydrophobic profile as placeholder
        for i, aa in enumerate(sequence):
            scores[i] = self.biophysics.hydrophobicity.get(aa, 0)

        # Min-max normalization to [0,1]
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            scores = (scores - score_min) / (score_max - score_min)
        else:
            scores = np.zeros_like(scores)

        return scores

    def _get_default_frustration_features(self) -> Dict[str, Any]:
        """
        Generate zero-valued default features for invalid/short sequences.

        Returns:
            Dict[str, Any]: Feature dictionary with all values set to zero
                           and zero-filled 100-dim frustration vector
        """
        return {
            'Frustration_NTerminal_Mean': 0.0,
            'Frustration_CTerminal_Mean': 0.0,
            'Frustration_Structural_Mean': 0.0,
            'Frustration_SignalVsStructure': 0.0,
            'Frustration_MaxInSignalRegion': 0.0,
            'Frustration_Localization': 0.0,
            'Frustration_HotspotCount': 0,
            'Frustration_SatisfactionRatio': 0.0,
            'Frustration_ProfileCorrelation': 0.0,
            'Frustration_HighSignalFrustration': 0.0,
            'Frustration_PerResidue_Vector': np.zeros(100, dtype=np.float32)
        }