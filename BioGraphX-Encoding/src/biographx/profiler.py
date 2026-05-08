"""
MotifProfiler
================================================================================

Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Implements comprehensive motif scanning and subcellular localization signal
    detection algorithms. This class systematically identifies peptide sequence
    patterns, structural motifs, and physicochemical signatures characteristic
    of distinct subcellular compartments.

    The profiler integrates two complementary approaches:
    1. Pattern-based detection using regular expressions for canonical
       localization signals (e.g., NLS, PTS1, KDEL)
    2. Biophysical property analysis (hydrophobic moment, charge distribution,
       transmembrane propensity) through dependency injection of BioPhysicsStrategy

    Ten major subcellular compartments are supported, each with organelle-specific
    scoring heuristics derived from empirical signal peptide grammars and
    retention motif databases.


Notes on Distance Metrics:
    IMPORTANT: Any distance calculations referenced in motif context refer to
    LINEAR SEQUENCE POSITIONS (residue indices), not 3D spatial measurements.
    This is particularly relevant when evaluating bipartite signals with gap
    lengths (e.g., NLS with 9-12 residue spacing), which represent primary
    structure separation, not tertiary structural proximity.

================================================================================
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from biographx.biophysics import BioPhysicsStrategy  # Assuming relative import


class MotifProfiler:
    """
    Sequence-based subcellular localization signal detector.

    This class provides organelle-specific scoring functions that evaluate
    amino acid sequences for the presence of canonical targeting motifs and
    physicochemical properties characteristic of different subcellular
    compartments. Each scoring method implements heuristic rules derived from
    experimental characterization of signal peptides and retention motifs.

    Attributes
    ----------
    biophysics : BioPhysicsStrategy
        Injected dependency providing biophysical property scales and
        calculation methods (hydrophobic moment, TMD score, etc.).

    Methods
    -------
    score_nucleus(seq)
        Detects mono/bipartite nuclear localization signals (NLS).
    score_mitochondria(seq)
        Evaluates mitochondrial targeting sequences (MTS) via amphipathic helix.
    score_extracellular(seq)
        Identifies classical secretory signal peptides (n-h-c pattern).
    score_membrane(seq)
        Predicts transmembrane domains via hydrophobicity scanning.
    score_er(seq)
        Detects ER retention/retrieval signals (KDEL, KKXX).
    score_golgi(seq)
        Identifies Golgi retention motifs (short TMD, tyrosine motifs).
    score_lysosome(seq)
        Recognizes lysosomal sorting signals (dileucine, tyrosine motifs).
    score_peroxisome(seq)
        Detects peroxisomal targeting signals PTS1 (C-terminal) and PTS2.
    score_plastid(seq)
        Evaluates chloroplast transit peptides (cTP) features.
    score_cytoplasm(seq)
        Default scoring based on absence of strong targeting signals.

    Notes
    -----
    All scoring functions return normalized values (0.0-1.0) representing
    confidence for localization to the respective compartment.
    """

    def __init__(self, biophysics_strategy: BioPhysicsStrategy):
        """
        Initialize MotifProfiler with a biophysics strategy instance.

        Parameters
        ----------
        biophysics_strategy : BioPhysicsStrategy
            Pre-configured instance containing hydrophobicity scales,
            pKa values, and biophysical calculation methods.
        """
        self.biophysics = biophysics_strategy

    # -------------------------------------------------------------------------
    # NUCLEAR LOCALIZATION SIGNALS (NLS)
    # -------------------------------------------------------------------------
    def score_nucleus(self, seq: str) -> float:
        """
        Detect nuclear localization signals (NLS) via pattern matching.

        Evaluates sequences for canonical monopartite and bipartite NLS
        patterns, as well as DNA-binding domain signatures.

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Nuclear localization confidence score (0.0-1.0):
            - 1.0: Bipartite NLS detected
            - 0.8: Monopartite NLS detected
            - 0.7: Zinc finger/DNA-binding motif detected
            - 0.0: No nuclear signals detected

        """
        score = 0.0

        # Monopartite NLS (e.g., PKKKRKV from SV40 T-antigen)
        # Pattern: cluster of 4+ basic residues, or Pro followed by 3+ basic residues
        if re.search(r'[RK]{4,}', seq) or re.search(r'P[RK]{3,}', seq):
            score = max(score, 0.8)

        # Bipartite NLS (e.g., nucleoplasmin: KR...10-12aa...KKKR)
        # IMPORTANT: Gap length represents LINEAR SEQUENCE DISTANCE, not 3D spacing
        if re.search(r'[RK]{2}.{9,12}[RK]{3}', seq):
            score = max(score, 1.0)

        # DNA-binding domain signature (e.g., C2H2 zinc finger)
        if re.search(r'C.{2,4}C.{12}H.{2,5}H', seq):
            score = max(score, 0.7)

        return score

    # -------------------------------------------------------------------------
    # MITOCHONDRIAL TARGETING SEQUENCES (MTS)
    # -------------------------------------------------------------------------
    def score_mitochondria(self, seq: str) -> float:
        """
        Evaluate mitochondrial targeting sequences (MTS).

        Mitochondrial presequences are characterized by N-terminal enrichment
        of positive charges and the ability to form amphipathic α-helices.

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Mitochondrial localization confidence score (0.0-1.0), derived from
            net positive charge and hydrophobic moment measurements.

        Notes
        -----
        Analysis focuses on first 25 residues where MTS are typically located.
        Combines charge score and amphipathic helix score.
        """
        if len(seq) < 20:
            return 0.0

        n_term = seq[:25]

        # 1. Net positive charge enrichment (characteristic of MTS)
        pos = sum(1 for aa in n_term if aa in 'RK')
        neg = sum(1 for aa in n_term if aa in 'DE')
        charge_score = 0.0
        if pos > 2 and neg < 2:  # High positive, low negative
            charge_score = 0.5

        # 2. Hydrophobic moment for α-helix (100° periodicity)
        # Measures amphipathicity - essential for mitochondrial import receptor recognition
        moment = self.biophysics.calculate_hydrophobic_moment(n_term)
        moment_score = min(moment / 10.0, 1.0)

        return (charge_score + moment_score) / 1.5

    # -------------------------------------------------------------------------
    # SECRETORY PATHWAY / EXTRACELLULAR
    # -------------------------------------------------------------------------
    def score_extracellular(self, seq: str) -> float:
        """
        Detect classical secretory signal peptides (n-h-c pattern).

        Implements the canonical signal peptide grammar:
        - n-region: N-terminal, often contains basic residues
        - h-region: Hydrophobic core (critical for SRP recognition)
        - c-region: Cleavage site with small/polar residues

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Secretory signal peptide confidence score (0.0-1.0).
            Returns 0.0 if hydrophobic core score < 0.6.

        """
        if len(seq) < 30:
            return 0.0

        n_region = seq[:5]
        h_region = seq[5:18]  # Hydrophobic core (typically 8-12 residues)
        c_region = seq[18:25]  # Cleavage region

        # n-region: Basic residues common
        n_score = 1.0 if any(aa in 'RK' for aa in n_region) else 0.0

        # h-region: Hydrophobic core (most discriminatory feature)
        h_count = sum(1 for aa in h_region if aa in 'LIVAFW')
        h_score = h_count / len(h_region)

        # c-region: Small residues at cleavage site (-1, -3 positions)
        c_score = 1.0 if any(aa in 'GAS' for aa in c_region) else 0.0

        # Hydrophobic core is mandatory; without it, no signal peptide
        if h_score > 0.6:
            return 0.4 * n_score + 0.4 * h_score + 0.2 * c_score
        return 0.0

    # -------------------------------------------------------------------------
    # PLASMA MEMBRANE / TRANSMEMBRANE
    # -------------------------------------------------------------------------
    def score_membrane(self, seq: str) -> float:
        """
        Predict transmembrane domain (TMD) containing proteins.

        Identifies potential integral membrane proteins through detection of
        long hydrophobic stretches characteristic of transmembrane helices.

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Membrane protein confidence score (0.0-1.0), derived from:
            - TMD score from sliding window hydrophobicity scan
            - Global hydrophobicity score for multi-pass proteins

        Notes
        -----
        This is a simplified predictor. Multi-pass proteins with multiple TMDs
        may also score highly due to elevated global hydrophobicity.
        """
        tmd_score = self.biophysics.calculate_tmd_score(seq)

        # Multi-pass proteins: elevated overall hydrophobicity
        global_hydro = sum(self.biophysics.hydrophobicity.get(aa, 0) for aa in seq) / len(seq)
        global_score = 1.0 if global_hydro > 0.5 else 0.0

        return max(tmd_score, global_score)

    # -------------------------------------------------------------------------
    # ENDOPLASMIC RETICULUM (ER)
    # -------------------------------------------------------------------------
    def score_er(self, seq: str) -> float:
        """
        Detect ER retention and retrieval signals.

        Identifies soluble ER resident proteins via C-terminal KDEL-like motifs
        and membrane ER proteins via C-terminal dilysine (KKXX) motifs.

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            ER localization confidence score (0.0-1.0):
            - 1.0: Strong C-terminal retention signal (KDEL, HDEL, RDEL)
            - 0.9: Dilysine motif (KKXX, KXKXX) for membrane proteins
            - 0.2 * signal_peptide_score: Secretory proteins default to ER


        """
        if len(seq) < 5:
            return 0.0

        c_term = seq[-5:]

        # Soluble ER resident proteins: C-terminal tetrapeptide retention signal
        if re.search(r'[KHR]DEL$', c_term):
            return 1.0

        # Membrane ER proteins: C-terminal dilysine motif (KKXX or KXKXX)
        if re.search(r'K[K\.].{2}$', c_term) or re.search(r'K.K.{2}$', c_term):
            return 0.9

        # Default ER entry: proteins entering secretory pathway via signal peptide
        sp_score = self.score_extracellular(seq)
        return 0.2 * sp_score

    # -------------------------------------------------------------------------
    # GOLGI APPARATUS
    # -------------------------------------------------------------------------
    def score_golgi(self, seq: str) -> float:
        """
        Detect Golgi retention signals.

        Golgi-resident membrane proteins typically have:
        1. Transmembrane domains shorter/less hydrophobic than plasma membrane TMDs
        2. Cytoplasmic tyrosine-based motifs (YXXΦ) for retrieval

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Golgi localization confidence score (0.0-1.0).

        Notes
        -----
        Differentiates from ER and plasma membrane by intermediate TMD
        hydrophobicity and absence of strong ER retention signals.
        """
        tmd_score = self.biophysics.calculate_tmd_score(seq)

        # Tyrosine-based sorting motif (YXXΦ, Φ = L,I,F,M,V)
        y_motif = 1.0 if re.search(r'Y..[LIFMV]', seq) else 0.0

        # Golgi TMDs: intermediate hydrophobicity (weaker than plasma membrane,
        # but still detectable as transmembrane)
        if 0.4 < tmd_score < 0.8:
            return 0.5 + (0.3 * y_motif)

        return 0.1 * y_motif

    # -------------------------------------------------------------------------
    # LYSOSOME / VACUOLE
    # -------------------------------------------------------------------------
    def score_lysosome(self, seq: str) -> float:
        """
        Detect lysosomal/vacuolar sorting signals.

        Recognizes two major classes of lysosomal targeting motifs:
        1. Dileucine-based signals [DE]XXXL[LI]
        2. Tyrosine-based signals YXXΦ

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Lysosomal localization confidence score (0.0-1.0).


        """
        score = 0.0

        # Dileucine-based sorting signal
        if re.search(r'[DE].{3}L[LI]', seq):
            score = max(score, 0.9)

        # Tyrosine-based sorting signal (YXXΦ)
        if re.search(r'GY..[LIFMV]', seq):
            score = max(score, 0.8)

        # N-glycosylation sites - lysosomal proteins are heavily glycosylated
        n_glyco = len(re.findall(r'N.[ST]', seq))
        glyco_score = min(n_glyco / 10.0, 0.5)

        return max(score, glyco_score)

    # -------------------------------------------------------------------------
    # PEROXISOME
    # -------------------------------------------------------------------------
    def score_peroxisome(self, seq: str) -> float:
        """
        Detect peroxisomal targeting signals (PTS1 and PTS2).

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Peroxisomal localization confidence score (0.0-1.0):
            - 1.0: Strong PTS1 (C-terminal SKL/AKL/SRL)
            - 0.9: PTS2 (N-terminal RLXXXXH/L pattern)
            - 0.0: No peroxisomal signals detected

        """
        if len(seq) < 3:
            return 0.0

        # PTS1: C-terminal tripeptide (most common: SKL, AKL, SRL)
        if re.search(r'[SA][KR][LM]$', seq[-3:]):
            return 1.0

        # PTS2: N-terminal nonapeptide (RLX5H/L)
        if len(seq) > 20 and re.search(r'RL.{5}[HL]', seq[:30]):
            return 0.9

        return 0.0

    # -------------------------------------------------------------------------
    # PLASTID (CHLOROPLAST)
    # -------------------------------------------------------------------------
    def score_plastid(self, seq: str) -> float:
        """
        Detect chloroplast transit peptides (cTP).

        Plastid targeting sequences are characterized by:
        - N-terminal enrichment of Ser/Thr residues
        - Depletion of acidic residues (Asp, Glu)
        - Unstructured, often low-complexity regions

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Plastid localization confidence score (0.0-1.0).


        """
        if len(seq) < 50:
            return 0.0

        n_term = seq[:50]

        # High Ser/Thr content (hallmark of chloroplast transit peptides)
        st_count = sum(1 for aa in n_term if aa in 'ST')

        # Low acidic residue content (negative charges disrupt import)
        de_count = sum(1 for aa in n_term if aa in 'DE')

        if st_count > 10 and de_count < 3:
            return 0.9

        return 0.0

    # -------------------------------------------------------------------------
    # CYTOPLASM
    # -------------------------------------------------------------------------
    def score_cytoplasm(self, seq: str) -> float:
        """
        Evaluate cytoplasmic localization probability.

        Cytoplasmic proteins are characterized by absence of strong targeting
        signals and balanced charge distribution. This serves as a default
        compartment when other signals are absent.

        Parameters
        ----------
        seq : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Cytoplasmic localization confidence score (0.0-1.0).

        Notes
        -----
        Exclusion-based scoring: membrane-targeting signals strongly
        contraindicate cytoplasmic localization.
        """
        # Exclusion: proteins with strong transmembrane domains are not cytoplasmic
        tmd = self.biophysics.calculate_tmd_score(seq)
        if tmd > 0.5:
            return 0.0

        # Charge balance between positive and negative residues
        pos = sum(1 for aa in seq if aa in 'KRH')
        neg = sum(1 for aa in seq if aa in 'DE')

        if pos + neg == 0:
            return 0.0

        balance = min(pos, neg) / max(pos, neg)
        return balance

    # -------------------------------------------------------------------------
    # ENHANCED PEROXISOMAL SCORING (DETAILED PATTERN ANALYSIS)
    # -------------------------------------------------------------------------
    def calculate_peroxisomal_residue_score(self, sequence: str) -> float:
        """
        Enhanced peroxisomal residue scoring with comprehensive pattern analysis.

        Implements a hierarchical scoring system for peroxisomal targeting signals
        with graduated confidence levels based on motif strength and position.

        Parameters
        ----------
        sequence : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Enhanced peroxisomal confidence score (0.0-1.0), considering:
            - Strong/weak PTS1 variants at C-terminus (weight: highest)
            - PTS2-like patterns at N-terminus
            - Internal peroxisomal motifs
            - C-terminal residue composition

        Notes
        -----
        This method provides finer discrimination between canonical and variant
        peroxisomal signals compared to the binary score_peroxisome() method.
        """
        seq = sequence.upper()
        scores = []

        # 1. Strong PTS1 signals at C-terminus (highest confidence)
        if len(seq) >= 3:
            last_3 = seq[-3:]
            # Canonical strong PTS1 signals
            if last_3 in ['SKL', 'AKL', 'SRL']:
                scores.append(1.0)
            # Moderate/weak PTS1 variants (still functional in some species)
            elif last_3 in ['SRM', 'ARL', 'PRL', 'NKL']:
                scores.append(0.8)
            # Minimal PTS1: XKL pattern
            elif last_3[1:] == 'KL':
                scores.append(0.6)

        # 2. PTS2-like signals (N-terminal region)
        if len(seq) >= 12:
            n_terminal = seq[:12]
            # RL dipeptide at positions 1-2
            if n_terminal[:2] == 'RL':
                # Check for additional basic residues in the spacer
                basic_count = sum(1 for aa in n_terminal[2:8] if aa in 'KRH')
                if basic_count >= 1:
                    scores.append(0.7)

        # 3. Internal peroxisomal motifs (lower confidence)
        internal_motifs = [
            r'RL..[QV]',  # PTS2-like internal variant
            r'RL..[HI]',  # PTS2-like with His/Ile
            r'[KR]..L',  # Basic-X-X-Leu pattern
        ]

        for pattern in internal_motifs:
            if re.search(pattern, seq):
                scores.append(0.4)
                break

        # 4. C-terminal region composition (last 20 residues)
        if len(seq) >= 20:
            c_region = seq[-20:]
            # Peroxisomal proteins enriched in small/basic residues at C-terminus
            small_residues = sum(1 for aa in c_region if aa in 'ASGT')
            basic_residues = sum(1 for aa in c_region if aa in 'KRH')
            composition_score = (small_residues + basic_residues) / 40  # Normalized
            scores.append(composition_score)

        # 5. Generic basic residue frequency (fallback)
        key_residues = set('SKLHRNQ')
        basic_score = sum(1 for aa in seq if aa in key_residues) / len(seq)
        scores.append(basic_score * 0.3)  # Lower weight for generic composition

        return max(scores) if scores else 0.0

    # -------------------------------------------------------------------------
    # ENHANCED PLASTID SCORING (DETAILED PATTERN ANALYSIS)
    # -------------------------------------------------------------------------
    def calculate_plastid_residue_score(self, sequence: str) -> float:
        """
        Enhanced plastid (chloroplast) scoring with comprehensive transit peptide analysis.

        Evaluates multiple features characteristic of chloroplast transit peptides
        (cTPs) including compositional bias, amphipathic patterns, and specific
        targeting motifs.

        Parameters
        ----------
        sequence : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Enhanced plastid confidence score (0.0-1.0).

        """
        seq = sequence.upper()
        scores = []

        # 1. N-terminal enrichment of small hydroxylated residues (Ser, Thr)
        if len(seq) >= 50:
            n_region = seq[:50]
            st_content = sum(1 for aa in n_region if aa in 'ST')
            st_score = st_content / 50
            scores.append(st_score)

            # Low acidic residue content in N-terminal (characteristic of cTP)
            acidic_content = sum(1 for aa in n_region if aa in 'DE')
            if acidic_content / 50 < 0.1:  # Less than 10% acidic residues
                scores.append(0.7)

        # 2. Amphipathic pattern in first 30 residues
        if len(seq) >= 30:
            first_30 = seq[:30]
            # Mixed hydrophobic/hydrophilic character (not strongly hydrophobic)
            hydrophobic = sum(1 for aa in first_30 if aa in 'AVILMFYW')
            hydrophilic = sum(1 for aa in first_30 if aa in 'STKRH')
            if hydrophobic > 8 and hydrophilic > 8:
                scores.append(0.6)

        # 3. Specific chloroplast targeting motifs
        plastid_motifs = [
            r'MA.{0,5}[AS]A',  # Met-Ala followed by small residues
            r'[VL]R.[AS]',  # Val/Leu-Arg-X-Ser/Ala
            r'[ST]..[RK]',  # Ser/Thr-X-X-Arg/Lys
        ]

        for pattern in plastid_motifs:
            if re.search(pattern, seq[:40]):  # Restricted to first 40 residues
                scores.append(0.5)
                break

        # 4. Low complexity regions (common in transit peptides)
        if len(seq) >= 30:
            first_30 = seq[:30]
            unique_residues = len(set(first_30))
            if unique_residues < 15:  # Low amino acid diversity
                scores.append(0.4)

        # 5. Generic enrichment of cTP-favored residues (fallback)
        key_residues = set('STAFLWYRK')
        basic_score = sum(1 for aa in seq if aa in key_residues) / len(seq)
        scores.append(basic_score * 0.4)

        return max(scores) if scores else 0.0

    # -------------------------------------------------------------------------
    # ENHANCED CYTOPLASMIC SCORING (COMPOSITIONAL ANALYSIS)
    # -------------------------------------------------------------------------
    def calculate_cytoplasm_residue_score(self, sequence: str) -> float:
        """
        Enhanced cytoplasmic scoring based on compositional and exclusionary features.

        Evaluates cytoplasmic localization probability through:
        - Absence of strong organelle-targeting signals
        - Balanced electrostatic charge
        - Moderate hydrophobicity
        - Enrichment of soluble protein characteristics

        Parameters
        ----------
        sequence : str
            Amino acid sequence to analyze.

        Returns
        -------
        float
            Enhanced cytoplasmic confidence score (0.0-1.0).

        Notes
        -----
        Cytoplasmic localization often serves as the "default" compartment;
        scoring emphasizes features consistent with soluble, non-secreted
        proteins rather than specific targeting motifs.
        """
        seq = sequence.upper()
        scores = []

        # 1. Absence of strong targeting signals in N-terminus
        if len(seq) > 20:
            hydrophobic_region = sum(1 for aa in seq[1:15] if aa in 'AVILMFYW')
            if hydrophobic_region < 8:  # Not a strong signal peptide
                scores.append(0.3)

        # 2. Balanced charge distribution
        basic = sum(1 for aa in seq if aa in 'KRH')
        acidic = sum(1 for aa in seq if aa in 'DE')
        total_charged = basic + acidic

        if total_charged > 0:
            charge_balance = min(basic, acidic) / max(basic, acidic) if max(basic, acidic) > 0 else 0
            scores.append(charge_balance * 0.5)

        # 3. Presence of common cytoplasmic motifs (loop regions, flexible turns)
        cytoplasm_motifs = [
            r'G[DE]',  # Gly-Asp/Glu (common in loops)
            r'P[ST]',  # Pro-Ser/Thr (flexible turns)
            r'[ED][ED][RK]',  # Acidic-acidic-basic clusters
        ]

        for pattern in cytoplasm_motifs:
            if re.search(pattern, seq):
                scores.append(0.3)
                break

        # 4. Moderate hydrophobicity (neither too high nor too low)
        hydrophobic = sum(1 for aa in seq if aa in 'AVILMFYW')
        hydrophobicity = hydrophobic / len(seq)

        # Cytoplasmic proteins typically have intermediate hydrophobicity
        if 0.3 <= hydrophobicity <= 0.5:
            scores.append(0.6)
        elif 0.25 <= hydrophobicity <= 0.55:
            scores.append(0.4)
        else:
            scores.append(0.1)

        # 5. Enrichment of charged/polar residues (soluble protein signature)
        charged_polar = sum(1 for aa in seq if aa in 'EDKHRQNSTY')
        cp_score = charged_polar / len(seq)
        scores.append(cp_score * 0.5)

        # 6. Depletion of rare targeting residues (e.g., Cys for disulfides)
        rare_targeting = sum(1 for aa in seq if aa in 'C')  # Cysteine for disulfide bonds
        if rare_targeting / len(seq) < 0.02:  # Less than 2% Cys
            scores.append(0.4)

        return np.mean(scores) if scores else 0.5  # Default to 0.5 if no signals

    # -------------------------------------------------------------------------
    # INTEGRATED FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    def extract_targeting_signal_features(self, sequence: str, hybrid_scores: Dict[str, float]) -> np.ndarray:
        """
        Generate integrated feature vector for subcellular localization prediction.

        Combines two orthogonal signal types for each of ten subcellular
        compartments:
        1. Biophysical/pattern-based scores (canonical targeting motifs)
        2. Hybrid interaction scores (graph-derived interaction enrichment)

        Parameters
        ----------
        sequence : str
            Amino acid sequence to analyze.
        hybrid_scores : Dict[str, float]
            Precomputed scores for each hybrid interaction type from graph analysis.
            Keys: hybrid interaction names, Values: normalized enrichment scores.

        Returns
        -------
        np.ndarray (dtype=np.float32)
            20-dimensional feature vector (2 features × 10 compartments):
            Even indices (0,2,4,...): Biophysical motif scores
            Odd indices (1,3,5,...): Hybrid interaction enrichment scores

        Notes
        -----
        This feature representation integrates sequence-based and interaction-based
        evidence for ensemble-based localization prediction. The vector is
        explicitly ordered to maintain compatibility with downstream classifiers.
        """
        features = []

        # Mapping of compartment names to their respective scoring functions
        scorers = {
            "Nucleus": self.score_nucleus,
            "Mitochondrion": self.score_mitochondria,
            "Extracellular": self.score_extracellular,
            "Cell.membrane": self.score_membrane,
            "Endoplasmic.reticulum": self.score_er,
            "Golgi.apparatus": self.score_golgi,
            "Lysosome/Vacuole": self.score_lysosome,
            "Peroxisomal": self.score_peroxisome,
            "Plastid": self.score_plastid,
            "Cytoplasm": self.score_cytoplasm
        }

        # Process each compartment in the order defined by localization_configs
        for loc, config in self.biophysics.localization_configs.items():
            # 1. Biophysical/Pattern-based score (sequence motifs)
            if loc in scorers:
                res_score = scorers[loc](sequence)
            else:
                res_score = 0.0
            features.append(res_score)

            # 2. Hybrid interaction enrichment score (graph-derived)
            hyb_list = [hybrid_scores[h] for h in config['expected_hybrids'] if h in hybrid_scores]
            hyb_score = np.mean(hyb_list) if hyb_list else 0.0
            features.append(hyb_score)

        result = np.array(features, dtype=np.float32)
        return result

    # Compatibility alias for legacy code expecting profile terminology
    extract_knowledge_profiles = extract_targeting_signal_features