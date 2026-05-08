"""
Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Core biophysical rule engine for protein sequence analysis. Implements comprehensive
    physicochemical property calculations, interaction rules, and structural feature
    extraction for computational proteomics.

    This module serves as the central repository for all biophysical constants and
    rules governing protein behavior, including:
    - Non-covalent interaction networks (hydrophobic, H-bond, salt bridges, etc.)
    - Amino acid physicochemical properties (hydrophobicity, charge, volume)
    - Secondary structure propensities
    - Hybrid interaction scoring for subcellular localization
"""

from math import sin, cos, radians, sqrt, log2
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BioPhysicsStrategy:
    """
    Central repository for biophysical rules, interaction constants, and physicochemical scales.

    This class encapsulates all domain-specific knowledge required for analyzing protein
    sequences, including interaction potentials, property scales, and feature extraction
    methodologies. All distances are measured in linear sequence units (amino acid positions)
    rather than 3D Ångström spatial coordinates.

    Attributes:
        interaction_rules (Dict): Parameters for 13 non-covalent interaction types with
            residue specificity, interaction strength, and maximum sequence separation.
        hybrid_interactions (Dict): Composite interactions combining two primary types
            with weighted scoring for synergistic effects.
        localization_configs (Dict): Expected hybrid interaction patterns for 10
            subcellular compartments.
        hydrophobicity (Dict): Kyte-Doolittle hydropathy scale (normalized).
        pka (Dict): Ionizable group pKa values for isoelectric point calculation.
        charge (Dict): Net charge per residue at physiological pH 7.0.
        aa_volume (Dict): Amino acid van der Waals volumes (Å³).
        helix_propensity (Dict): Alpha-helix formation propensities normalized to Pα=1.0.
    """

    def __init__(self):
        """
        Initialize biophysical rule sets and physicochemical property scales.

        All distance thresholds represent maximum sequence separation (number of
        intervening residues) for considering interactions, not spatial proximity
        in 3D conformation.
        """
        # 1. GRAPH RULES (Standard)
        # Distance thresholds are linear sequence positions, NOT 3D Ångström measurements
        self.interaction_rules = {
            'hydrophobic': {
                'residues': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y', 'C'],
                'strength': 1.0,
                'max_distance': 20  # Linear sequence positions
            },
            'hydrogen_bond': {
                'donors': ['N', 'Q', 'S', 'T', 'Y', 'H', 'K', 'R', 'W'],
                'acceptors': ['D', 'E', 'N', 'Q', 'S', 'T', 'Y'],
                'strength': 0.8,
                'max_distance': 12  # Linear sequence positions
            },
            'salt_bridge': {
                'positive': ['K', 'R', 'H'],
                'negative': ['D', 'E'],
                'strength': 0.7,
                'max_distance': 35  # Linear sequence positions
            },
            'disulfide': {
                'residues': ['C'],
                'strength': 0.9,
                'max_distance': 2000  # Linear positions; covalent bond independent of distance
            },
            'pi_interaction': {
                'aromatic': ['F', 'Y', 'W'],
                'positive': ['R', 'K', 'H'],
                'strength': 0.6,
                'max_distance': 15  # Linear sequence positions
            },
            'cation_pi': {
                'positive': ['K', 'R', 'H'],
                'aromatic': ['F', 'Y', 'W'],
                'strength': 0.65,
                'max_distance': 10  # Linear sequence positions
            },
            'van_der_waals': {
                'residues': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y', 'C', 'P'],
                'strength': 0.4,
                'max_distance': 6  # Linear sequence positions
            },
            'ch_pi': {
                'donors': ['A', 'V', 'L', 'I', 'M', 'C'],
                'acceptors': ['F', 'Y', 'W'],
                'strength': 0.3,
                'max_distance': 8  # Linear sequence positions
            },
            'nh_pi': {
                'donors': ['N', 'Q', 'S', 'T', 'H', 'K', 'R', 'W'],
                'acceptors': ['F', 'Y', 'W'],
                'strength': 0.5,
                'max_distance': 8  # Linear sequence positions
            },
            'carbonyl_carbonyl': {
                'residues': ['D', 'E', 'N', 'Q', 'S', 'T', 'Y'],
                'strength': 0.35,
                'max_distance': 8  # Linear sequence positions
            },
            'sulfur_pi': {
                'sulfur_residues': ['M', 'C'],
                'aromatic': ['F', 'Y', 'W'],
                'strength': 0.45,
                'max_distance': 8  # Linear sequence positions
            },
            'backbone': {
                'residues': [],
                'strength': 1.0,
                'max_distance': 1  # Linear sequence positions; adjacent residues
            }
        }

        # 2. HYBRID INTERACTIONS
        # Composite interactions capturing cooperative effects between primary and secondary types
        self.hybrid_interactions = {
            'salt_bridge_hbond': {'primary': 'salt_bridge', 'secondary': 'hydrogen_bond', 'weight': 1.2},
            'hydrophobic_pi': {'primary': 'hydrophobic', 'secondary': 'pi_interaction', 'weight': 1.1},
            'cation_pi_hbond_network': {'primary': 'cation_pi', 'secondary': 'hydrogen_bond', 'weight': 1.15},
            'hydrophobic_vdw_cluster': {'primary': 'hydrophobic', 'secondary': 'van_der_waals', 'weight': 1.05},
            'pi_cation_hbond': {'primary': 'pi_interaction', 'secondary': 'hydrogen_bond', 'weight': 1.1},
            'ch_pi_hydrophobic': {'primary': 'ch_pi', 'secondary': 'hydrophobic', 'weight': 1.0},
            'sulfur_aromatic_network': {'primary': 'sulfur_pi', 'secondary': 'pi_interaction', 'weight': 1.2},
            'carbonyl_charge_cluster': {'primary': 'carbonyl_carbonyl', 'secondary': 'salt_bridge', 'weight': 1.1}
        }

        # 3. LOCALIZATION CONFIGS
        # Subcellular compartment-specific enrichment patterns for hybrid interactions
        self.localization_configs = {
            "Nucleus": {'expected_hybrids': ['salt_bridge_hbond', 'cation_pi_hbond_network', 'pi_cation_hbond']},
            "Mitochondrion": {'expected_hybrids': ['salt_bridge_hbond', 'hydrophobic_pi']},
            "Extracellular": {'expected_hybrids': ['hydrophobic_pi', 'hydrophobic_vdw_cluster']},
            "Cell.membrane": {'expected_hybrids': ['hydrophobic_pi', 'hydrophobic_vdw_cluster', 'ch_pi_hydrophobic']},
            "Endoplasmic.reticulum": {'expected_hybrids': ['carbonyl_charge_cluster']},
            "Golgi.apparatus": {'expected_hybrids': ['hydrophobic_pi', 'sulfur_aromatic_network']},
            "Lysosome/Vacuole": {'expected_hybrids': ['salt_bridge_hbond', 'carbonyl_charge_cluster']},
            "Peroxisomal": {
                'expected_hybrids': ['sulfur_aromatic_network', 'carbonyl_charge_cluster']},
            "Plastid": {'expected_hybrids': ['hydrogen_bond', 'hydrophobic_pi', 'salt_bridge_hbond']},
            "Cytoplasm": {'expected_hybrids': ['salt_bridge_hbond', 'hydrogen_bond', 'hydrophobic_vdw_cluster']}
        }

        # 4. HYDROPHOBICITY SCALE (Kyte-Doolittle)
        # Normalized hydropathy index; positive = hydrophobic, negative = hydrophilic
        self.hydrophobicity = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
            'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
            'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }

        # 5. pKa VALUES
        # Acid dissociation constants for ionizable groups
        self.pka = {
            'COOH': 3.6,  # C-terminal carboxyl
            'NH2': 8.6,  # N-terminal amino
            'D': 3.9,  # Aspartic acid side chain
            'E': 4.1,  # Glutamic acid side chain
            'H': 6.5,  # Histidine side chain
            'C': 8.5,  # Cysteine side chain
            'Y': 10.1,  # Tyrosine side chain
            'K': 10.8,  # Lysine side chain
            'R': 12.5  # Arginine side chain
        }

        # 6. CHARGE TABLE (at pH 7.0)
        # Net electrostatic charge under physiological conditions
        self.charge = {
            'K': 1, 'R': 1, 'H': 0.1,  # Partially protonated histidine
            'D': -1, 'E': -1,
            'others': 0
        }

        # 7. AMINO ACID VOLUME (Å³)
        # van der Waals volumes for steric considerations
        self.aa_volume = {
            'G': 60.1, 'A': 88.6, 'S': 89.0, 'C': 108.5, 'D': 111.1, 'P': 112.7, 'N': 114.1,
            'T': 116.1, 'V': 140.0, 'E': 138.4, 'Q': 143.8, 'H': 153.2, 'L': 166.7, 'I': 166.7,
            'M': 162.9, 'K': 168.6, 'F': 189.9, 'R': 173.4, 'Y': 193.6, 'W': 227.8
        }

        # 8. HELIX PROPENSITY
        # Normalized frequency of alpha-helix formation (Pα)
        self.helix_propensity = {
            'A': 1.42, 'L': 1.21, 'M': 1.45, 'K': 1.16, 'E': 1.51, 'Q': 1.11, 'H': 1.00,
            'R': 0.98, 'F': 1.13, 'Y': 0.69, 'W': 1.08, 'I': 1.08, 'V': 1.06, 'T': 0.83,
            'S': 0.77, 'C': 0.70, 'D': 1.01, 'N': 0.67, 'P': 0.57, 'G': 0.57
        }

    def check_interaction(self, aa1: str, aa2: str, interaction_type: str) -> bool:
        """
        Determine if two amino acid residues can form a specific non-covalent interaction.

        Evaluates compatibility based on residue identity and interaction-specific rules.
        Distance validation (sequence separation) must be performed by the caller.

        Args:
            aa1: Single-letter amino acid code for first residue
            aa2: Single-letter amino acid code for second residue
            interaction_type: Type of interaction to check (key in interaction_rules)

        Returns:
            bool: True if residues satisfy chemical compatibility for the interaction

        Raises:
            KeyError: If interaction_type not found in interaction_rules
        """
        rules = self.interaction_rules[interaction_type]

        if interaction_type == 'salt_bridge':
            return (aa1 in rules['positive'] and aa2 in rules['negative']) or \
                (aa1 in rules['negative'] and aa2 in rules['positive'])

        elif interaction_type == 'hydrogen_bond':
            donors = set(rules['donors'])
            acceptors = set(rules['acceptors'])
            return (aa1 in donors and aa2 in acceptors) or \
                (aa1 in acceptors and aa2 in donors)

        elif interaction_type == 'hydrophobic':
            return aa1 in rules['residues'] and aa2 in rules['residues']

        elif interaction_type == 'pi_interaction':
            aromatic = set(rules['aromatic'])
            positive = set(rules['positive'])
            return (aa1 in aromatic and aa2 in aromatic) or \
                (aa1 in aromatic and aa2 in positive) or \
                (aa1 in positive and aa2 in aromatic)

        elif interaction_type == 'disulfide':
            return aa1 in rules['residues'] and aa2 in rules['residues']

        elif interaction_type == 'cation_pi':
            positive = set(rules['positive'])
            aromatic = set(rules['aromatic'])
            return (aa1 in positive and aa2 in aromatic) or \
                (aa1 in aromatic and aa2 in positive)

        elif interaction_type == 'van_der_waals':
            return aa1 in rules['residues'] and aa2 in rules['residues']

        elif interaction_type == 'ch_pi':
            donors = set(rules['donors'])
            acceptors = set(rules['acceptors'])
            return (aa1 in donors and aa2 in acceptors) or \
                (aa1 in acceptors and aa2 in donors)

        elif interaction_type == 'nh_pi':
            donors = set(rules['donors'])
            acceptors = set(rules['acceptors'])
            return (aa1 in donors and aa2 in acceptors) or \
                (aa1 in acceptors and aa2 in donors)

        elif interaction_type == 'carbonyl_carbonyl':
            return aa1 in rules['residues'] and aa2 in rules['residues']

        elif interaction_type == 'sulfur_pi':
            sulfur_residues = set(rules['sulfur_residues'])
            aromatic = set(rules['aromatic'])
            return (aa1 in sulfur_residues and aa2 in aromatic) or \
                (aa1 in aromatic and aa2 in sulfur_residues)

        return False

    def calculate_hydrophobic_moment(self, sequence: str, angle: float = 100) -> float:
        """
        Compute Eisenberg hydrophobic moment for amphipathicity assessment.

        The hydrophobic moment quantifies the asymmetry of hydrophobicity distribution
        around a helical wheel projection. Higher values indicate stronger amphipathicity.

        Args:
            sequence: Amino acid sequence (single-letter codes)
            angle: Rotation angle per residue in degrees (100° for alpha-helix)

        Returns:
            float: Magnitude of hydrophobic moment vector
        """
        h_vals = [self.hydrophobicity.get(aa, 0) for aa in sequence]
        sum_sin = sum(h * sin(radians(angle * i)) for i, h in enumerate(h_vals))
        sum_cos = sum(h * cos(radians(angle * i)) for i, h in enumerate(h_vals))
        return sqrt(sum_sin ** 2 + sum_cos ** 2)

    def calculate_tmd_score(self, sequence: str) -> float:
        """
        Predict transmembrane domain propensity via sliding window hydrophobicity.

        Uses 18-residue window scanning to identify potential membrane-spanning segments.
        Normalized to [0,1] range where higher scores indicate stronger TM likelihood.

        Args:
            sequence: Amino acid sequence (single-letter codes)

        Returns:
            float: Normalized transmembrane domain score (0.0-1.0)
        """
        window = 18
        max_score = 0
        if len(sequence) < window:
            return 0

        h_scores = [self.hydrophobicity.get(aa, -2) for aa in sequence]
        for i in range(len(h_scores) - window):
            segment_score = sum(h_scores[i:i + window])
            if segment_score > max_score:
                max_score = segment_score

        # Normalize to [0,1] scale (empirical maximum ~70)
        return min(max_score / 60.0, 1.0)

    def calculate_isoelectric_point(self, seq: str, tolerance: float = 0.01) -> float:
        """
        Compute theoretical isoelectric point (pI) via Henderson-Hasselbalch equation.

        Uses bisection method to find pH where net charge equals zero, accounting for
        N-terminus, C-terminus, and ionizable side chains.

        Args:
            seq: Amino acid sequence (single-letter codes)
            tolerance: Convergence criterion for bisection method (default: 0.01 pH units)

        Returns:
            float: Theoretical isoelectric point (pH)
        """
        counts = Counter(seq)

        def get_charge(pH: float) -> float:
            """
            Calculate net charge at specified pH.

            Args:
                pH: Hydrogen ion concentration exponent

            Returns:
                float: Net electrostatic charge
            """
            # N-terminal amino group
            charge = 1 / (1 + 10 ** (pH - self.pka['NH2']))
            # C-terminal carboxyl group
            charge -= 1 / (1 + 10 ** (self.pka['COOH'] - pH))

            # Side chain ionizable groups
            for aa, pK in self.pka.items():
                if aa in ['NH2', 'COOH']:
                    continue
                count = counts.get(aa, 0)
                if count == 0:
                    continue

                if aa in ['K', 'R', 'H']:  # Basic residues
                    charge += count / (1 + 10 ** (pH - pK))
                else:  # Acidic residues (D, E, C, Y)
                    charge -= count / (1 + 10 ** (pK - pH))
            return charge

        # Bisection method search over pH range [0,14]
        min_pH, max_pH = 0.0, 14.0
        while (max_pH - min_pH) > tolerance:
            mid_pH = (min_pH + max_pH) / 2
            charge = get_charge(mid_pH)
            if charge > 0:
                min_pH = mid_pH
            else:
                max_pH = mid_pH
        return (min_pH + max_pH) / 2

    def calculate_autocorrelation(self, seq: str, prop_map: Dict, lag_max: int = 10) -> List[float]:
        """
        Compute sequence autocorrelation for physicochemical property periodicity.

        Measures correlation of property values at positions i and i+lag, revealing
        periodic patterns relevant to secondary structure and functional motifs.

        Args:
            seq: Amino acid sequence (single-letter codes)
            prop_map: Property values keyed by amino acid
            lag_max: Maximum sequence separation to evaluate

        Returns:
            List[float]: Autocorrelation coefficients for lags 1 through lag_max
        """
        n = len(seq)
        if n < lag_max + 1:
            return [0.0] * lag_max

        vals = [prop_map.get(aa, 0) for aa in seq]
        mean_val = np.mean(vals)
        vals_centered = [x - mean_val for x in vals]
        variance = np.var(vals) if np.var(vals) > 0 else 1.0

        correlations = []
        for lag in range(1, lag_max + 1):
            s = sum(vals_centered[i] * vals_centered[i + lag] for i in range(n - lag))
            norm = (n - lag) * variance
            correlations.append(s / norm if norm > 0 else 0.0)
        return correlations

    def calculate_shannon_entropy(self, sequence: str) -> float:
        """
        Compute Shannon entropy of amino acid composition.

        Measures sequence diversity/complexity; higher values indicate more uniform
        residue distribution, lower values indicate compositional bias.

        Args:
            sequence: Amino acid sequence (single-letter codes)

        Returns:
            float: Shannon entropy (bits)
        """
        n = len(sequence)
        if n == 0:
            return 0.0
        counts = Counter(sequence)
        return -sum((c / n) * log2(c / n) for c in counts.values())

    def extract_global_physics(self, sequence: str) -> np.ndarray:
        """
        Extract comprehensive global physicochemical feature vector.

        Generates 19-dimensional feature vector encoding:
        - Isoelectric point, net charge, normalized charge density
        - GRAVY hydrophobicity index, aromaticity, instability proxy
        - Hydrophobicity and charge autocorrelation (lags 1-6)
        - Shannon entropy of composition

        Args:
            sequence: Amino acid sequence (single-letter codes)

        Returns:
            np.ndarray: 19-element float32 array of normalized features
        """
        features = []
        n = len(sequence)

        # 1. Isoelectric Point (pI)
        pi = self.calculate_isoelectric_point(sequence)
        features.append(pi)

        # 2. Electrostatic properties at pH 7.0
        pos = sum(1 for aa in sequence if aa in 'KRH')
        neg = sum(1 for aa in sequence if aa in 'DE')
        net_charge = pos - neg
        features.extend([net_charge, net_charge / n if n > 0 else 0])

        # 3. GRAVY (Grand Average of Hydropathicity)
        # Index of overall solubility/hydrophobicity
        total_hydro = sum(self.hydrophobicity.get(aa, 0) for aa in sequence)
        gravy = total_hydro / n if n > 0 else 0
        features.append(gravy)

        # 4. Aromaticity frequency
        arom = sum(1 for aa in sequence if aa in 'FYW')
        features.append(arom / n if n > 0 else 0)

        # 5. Instability proxy (PEST domain residues)
        # Residues associated with protein degradation signals
        instability = sum(1 for aa in sequence if aa in 'PESTQD')
        features.append(instability / n if n > 0 else 0)

        # 6. Hydrophobicity autocorrelation (periodicity analysis)
        hydro_corr = self.calculate_autocorrelation(sequence, self.hydrophobicity, lag_max=6)
        features.extend(hydro_corr)

        # 7. Charge autocorrelation (lags 1-6)
        charge_map = {aa: self.charge.get(aa, 0) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        charge_corr = self.calculate_autocorrelation(sequence, charge_map, lag_max=6)
        features.extend(charge_corr)

        # 8. Compositional entropy
        entropy = self.calculate_shannon_entropy(sequence)
        features.append(entropy)

        # Ensure consistent feature dimension (19 features)
        if len(features) < 19:
            features.extend([0.0] * (19 - len(features)))

        return np.array(features, dtype=np.float32)