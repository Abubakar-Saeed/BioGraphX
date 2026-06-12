"""
BioGraphX Targeting Rules and Canonical Motifs Database
Derived from empirical rule-sets and verified primary literature.

NOTE: This file documents the regular expressions and heuristic rules
that are ACTUALLY implemented in the BioGraphX encoding pipeline
(MotifProfiler class). Only patterns present in the code are listed.
"""

TARGETING_RULES = {
    "Nucleus": {
        "heuristics": {
            "monopartite_nls": r"[RK]{4,}",
            "proline_cluster": r"P[RK]{3,}",
            "bipartite_nls": r"[RK]{2}.{9,12}[RK]{3}",
            "zinc_finger": r"C.{2,4}C.{12}H.{2,5}H"
        },
        "description": "Detects classical mono/bipartite nuclear localization signals (NLS) and DNA-binding zinc finger domains.",
        "hybrid_indicators": ["salt_bridge_hbond", "cation_pi_hbond_network", "pi_cation_hbond"],
        "literature_citations": [
            "Kalderon, D., Roberts, B. L., Richardson, W. D., & Smith, A. E. (1984). A short amino acid sequence able to specify nuclear location. Cell, 39(3), 499-509.",
            "Robbins, J., Dilworth, S. M., Laskey, R. A., & Dingwall, C. (1991). Two interdependent basic domains in nucleoplasmin nuclear targeting sequence: identification of a formal bipartite signal. Cell, 64(3), 615-623.",
            "Miller, J., McLachlan, A. D., & Klug, A. (1985). Repetitive zinc-binding domains in the protein transcription factor IIIA from Xenopus oocytes. The EMBO Journal, 4(6), 1609-1614."
        ]
    },
    "Mitochondrion": {
        "heuristics": {
            "charge_window_length": 25,
            "min_net_positive_charge": ">2",   # code uses >2, equivalent to >=3
            "max_negative_charge": "<2",
            "amphipathic_propensity": "hydrophobic_moment > threshold"
        },
        "description": "Evaluates N-terminal mitochondrial targeting sequences (MTS) using net charge filtration and amphipathic alpha-helix propensities.",
        "hybrid_indicators": ["salt_bridge_hbond", "hydrophobic_pi"],
        "literature_citations": [
            "von Heijne, G. (1986). Mitochondrial targeting sequences may form amphiphilic helices. The EMBO Journal, 5(6), 1335-1342.",
            "Wiedemann, N., & Pfanner, N. (2017). Mitochondrial machineries for protein import and assembly. Annual Review of Biochemistry, 86, 685-714."
        ]
    },
    "Extracellular": {
        "heuristics": {
            "n_region_window": 5,
            "n_region_min_basic": 1,
            "h_region_window": (6, 18),
            "h_region_min_hydrophobic_prop": 0.60,
            "c_region_window": (19, 25),
            "c_region_allowed_cleavage": ["G", "A", "S"]
        },
        "description": "Classical secretory signal peptides defined by von Heijne's tripartite n-, h-, and c-region grammars.",
        "hybrid_indicators": ["hydrophobic_pi", "disulfide_hbond", "hydrophobic_vdw_cluster"],
        "literature_citations": [
            "von Heijne, G. (1985). Signal sequences: the limits of variation. Journal of Molecular Biology, 184(1), 99-105.",
            "Akopian, D., Shen, K., Zhang, X., & Shan, S. O. (2013). Signal recognition particle: an essential protein-targeting machine. Annual Review of Biochemistry, 82, 693-721."
        ]
    },
    "Cell.membrane": {
        "heuristics": {
            "sliding_window_len": 18,
            "hydrophobicity_scale": "Kyte-Doolittle",
            "tmd_threshold": "TMD score > threshold"
        },
        "description": "Predicts integral transmembrane domains (TMD) using continuous hydrophobic segment scanning.",
        "hybrid_indicators": ["hydrophobic_pi", "hydrophobic_vdw_cluster", "ch_pi_hydrophobic"],
        "literature_citations": [
            "Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. Journal of Molecular Biology, 157(1), 105-132.",
            "Rost, B., Fariselli, P., & Casadio, R. (1996). Topology prediction for helical transmembrane proteins at 86% accuracy. Protein Science, 5(8), 1704-1718."
        ]
    },
    "Endoplasmic.reticulum": {
        "heuristics": {
            "soluble_retention_regex": r"[KHR]DEL$",
            "membrane_retention_regex": r"K[K\.].{2}$|K.K.{2}$"
        },
        "description": "Scans for C-terminal luminal retention signals (KDEL family) and membrane-bound dilysine motifs (KKXX family).",
        "hybrid_indicators": ["disulfide_hbond", "carbonyl_charge_cluster"],
        "literature_citations": [
            "Munro, S., & Pelham, H. R. (1987). A C-terminal signal prevents secretion of luminal ER proteins. Cell, 48(6), 899-907.",
            "Nilsson, T., Jackson, M., & Peterson, P. A. (1989). Short cytoplasmic sequences serve as retention signals for transmembrane proteins in the endoplasmic reticulum. Cell, 58(4), 707-718."
        ]
    },
    "Golgi.apparatus": {
        "heuristics": {
            "tmd_bounds": [0.4, 0.8],
            "tyrosine_motif": r"Y..[LIFMV]"
        },
        "description": "Identifies Golgi-resident profiles via shorter/weaker transmembrane anchors and tyrosine-based cargo sorting motifs.",
        "hybrid_indicators": ["hydrophobic_pi", "sulfur_aromatic_network"],
        "literature_citations": [
            "Tu, L., & Banfield, D. K. (2010). Localization of Golgi-resident glycosyltransferases. Cellular and Molecular Life Sciences, 67(1), 29-41.",
            "Breuza, S., Halbeisen, R., Jenö, P., Otte, S., Barlowe, C., & Hong, W. (2002). Transmembrane domain length and sequence patterns govern Golgi retention. Journal of Cell Science, 115(23), 4457-4467."
        ]
    },
    "Lysosome/Vacuole": {
        "heuristics": {
            "dileucine_motif": r"[DE].{3}L[LI]",
            "tyrosine_motif": r"GY..[LIFMV]",
            "glycosylation_regex": r"N.[ST]"
        },
        "description": "Detects dileucine-based ([DE]XXXL[LI]) and tyrosine-based (GYXXØ) clathrin adaptor sorting signals along with N-glycosylation profiles.",
        "hybrid_indicators": ["salt_bridge_hbond", "carbonyl_charge_cluster"],
        "literature_citations": [
            "Bonifacino, J. S., & Traub, L. M. (2003). Signals for sorting of transmembrane proteins to endosomes and lysosomes. Annual Review of Biochemistry, 72(1), 395-447.",
            "Alberts, B., Johnson, A., Lewis, J., Raff, M., Roberts, K., & Walter, P. (2002). Molecular Biology of the Cell. Garland Science. Chapter: Protein Sorting."
        ]
    },
    "Peroxisomal": {
        "heuristics": {
            "pts1_terminal_regex": r"[SA][KR][LM]$",
            "pts2_nterm_regex": r"RL.{5}[HL]",
            # Internal PTS2 patterns and composition bias are NOT implemented in code
            # (only documented for future extension)
        },
        "description": "Identifies C-terminal Peroxisomal Targeting Signal 1 (PTS1) and N-terminal PTS2 motifs.",
        "hybrid_indicators": ["sulfur_aromatic_network", "disulfide_hbond", "carbonyl_charge_cluster"],
        "literature_citations": [
            "Gould, S. J., Keller, G. A., Schneider, N., Howell, S. H., & Subramani, S. (1989). Peroxisomal protein import is directed by a selective sequence with the terminal tripeptide as the primary determinant. Journal of Cell Biology, 108(5), 1657-1664.",
            "Osumi, T., Tsukamoto, T., Hata, S., Yokota, S., Miura, S., Fujiki, Y., & Hijikata, M. (1991). Amino-terminal region of rat kidney peroxisomal 3-ketoacyl-CoA thiolase contains a peroxisomal targeting signal (PTS2). Biochemical and Biophysical Research Communications, 181(3), 1138-1144."
        ]
    },
    "Plastid": {
        "heuristics": {
            "nterm_window": 50,
            "min_ser_thr_prop": 0.20,
            "max_acidic_prop": 0.10,
            "motifs_first_40_aa": [r"MA.{0,5}[AS]A", r"[VL]R.[AS]", r"[ST]..[RK]"]
            # Low complexity check (max_unique_residues) is NOT implemented
        },
        "description": "Evaluates Chloroplast Transit Peptides (cTP) through Ser/Thr enrichment, low acidic content, and specific N-terminal motifs.",
        "hybrid_indicators": ["hydrogen_bond", "hydrophobic_pi", "salt_bridge_hbond"],
        "literature_citations": [
            "Bruce, B. D. (2000). Chloroplast transit peptides: structure, function and evolution. Trends in Cell Biology, 10(10), 440-447.",
            "McFadden, G. I. (2014). Origin and evolution of plastids and photosynthesis in eukaryotes. Cold Spring Harbor Perspectives in Biology, 6(4), a016105.",
            "Schleiff, E., & Becker, T. (2011). Common ground for protein translocation: access control for mitochondria and chloroplasts. Nature Reviews Molecular Cell Biology, 12(1), 48-59."
        ]
    },
    "Cytoplasm": {
        "heuristics": {
            "nterm_hydrophobic_exclusion_window": 15,
            "max_hydrophobic_residues_exclusion": 7,
            "solubility_hydrophobicity_range": (0.3, 0.5),
            "max_cysteine_prop": 0.02,
            "loop_motifs": [r"G[DE]", r"P[ST]", r"[ED][ED][RK]"]
        },
        "description": "Default exclusionary and baseline verification architecture tracking balanced charges, absence of export signal structures, high structural solubility, and low cysteine occurrence.",
        "hybrid_indicators": ["clustering_coefficient", "hydro_corr_l6", "charge_density_physics"],
        "literature_citations": [
            "Alberts, B., Johnson, A., Lewis, J., Raff, M., Roberts, K., & Walter, P. (2002). Molecular Biology of the Cell. Garland Science. Chapter: The Cytoplasm and Cytoskeleton."
        ]
    }
}