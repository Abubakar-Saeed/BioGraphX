"""
Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Graph-theoretic protein sequence analysis engine for constructing and analyzing
    residue interaction networks. Implements constraint graph construction from
    biophysical interaction rules, extracting topological features relevant to
    protein structure and function.

    This module transforms linear amino acid sequences into weighted graph representations
    where vertices represent residues and edges represent potential non-covalent
    interactions. All distance constraints in graph construction are based on linear
    sequence separation (amino acid positions), NOT 3D spatial coordinates.

"""

import igraph as ig
import numpy as np
import warnings
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from biographx.biophysics import BioPhysicsStrategy
from biographx.profiler import MotifProfiler


class GraphEngine:
    """
    Protein residue interaction network constructor and graph feature extractor.

    This engine builds constraint graphs from protein sequences based on biophysical
    interaction rules. Graphs represent potential non-covalent interactions between
    residues, with vertices corresponding to sequence positions and edges encoding
    interaction types, strengths, and hybrid interaction classifications.

    All distance thresholds in edge construction are based on linear sequence
    separation (number of intervening residues), NOT 3D Ångström spatial coordinates.
    This sequence-based distance metric enables structure-independent analysis.

    Attributes:
        biophysics (BioPhysicsStrategy): Biophysical rule repository providing
            interaction rules, strength coefficients, and distance thresholds.
        motif_profiler (MotifProfiler): Sequence motif detector for functional
            pattern enrichment in graph regions.
    """

    def __init__(self, biophysics_strategy: BioPhysicsStrategy, motif_profiler: MotifProfiler):
        """
        Initialize graph engine with biophysical rules and motif profiling capabilities.

        Args:
            biophysics_strategy: Centralized biophysical constants and interaction rules
            motif_profiler: Sequence motif detection engine for functional annotation
        """
        self.biophysics = biophysics_strategy
        self.motif_profiler = motif_profiler

    def build_complete_graph(self, sequence: str) -> Tuple[ig.Graph, Dict[str, float]]:
        """
        Construct comprehensive residue interaction graph with hybrid edge tracking.

        Builds an undirected graph where vertices represent amino acid positions
        and edges represent potential biochemical interactions. Edge construction
        respects interaction-type specific distance thresholds measured in linear
        sequence positions. Includes both side-chain interactions and peptide backbone.

        Args:
            sequence: Input protein sequence (single-letter codes, case-insensitive)

        Returns:
            Tuple[ig.Graph, Dict[str, float]]:
                - Graph object with vertices annotated with residue identity/position
                - Dictionary mapping hybrid interaction types to normalized scores

        Edge Attributes:
            interaction_type: Dominant interaction type (e.g., 'hydrophobic', 'salt_bridge')
            weight: Normalized interaction strength [0,1] with distance decay
            interaction_set: Frozenset of all valid interaction types for this residue pair
            is_hybrid: Binary flag (1 if edge qualifies as hybrid interaction)

        Notes:
            - Optimization: Pairs with distance >35 only checked for disulfide bonds
            - Distance decay: weight *= (1 - 0.3 * distance/max_distance)
            - Hybrid weighting: Primary-secondary combinations receive boost factor
            - Backbone edges (i,i+1) added with unit weight and 'backbone' type
        """
        n = len(sequence)
        seq = sequence.upper()

        # Initialize graph with residue vertices
        graph = ig.Graph(n, directed=False)
        graph.vs["residue"] = list(seq)
        graph.vs["position"] = list(range(n))
        graph.vs["is_key"] = [0] * n  # Placeholder for motif-identified key residues

        # Edge collection containers
        edges = []
        edge_attributes = {
            "interaction_type": [],
            "weight": [],
            "interaction_set": [],
            "is_hybrid": []
        }

        # Track per-residue interaction types for hybrid analysis
        residue_interactions = defaultdict(set)
        hybrid_edge_counts = {hybrid_type: 0 for hybrid_type in self.biophysics.hybrid_interactions.keys()}

        # ---------------------------------------------------------
        # 1. SIDE CHAIN INTERACTIONS (Variable Distance)
        #    Distance thresholds are LINEAR SEQUENCE POSITIONS
        # ---------------------------------------------------------
        for i in range(n):
            # Scan forward to avoid duplicate edges
            for j in range(i + 1, n):
                distance = j - i  # Linear sequence separation

                # OPTIMIZATION: Early exit for long-range pairs
                # Only disulfide bonds can form at distances >35 residues
                if distance > 35:
                    if seq[i] != 'C' or seq[j] != 'C':
                        continue

                aa1, aa2 = seq[i], seq[j]

                # Identify all biophysically valid interactions for this pair
                found_interactions = set()
                for interaction_type, rules in self.biophysics.interaction_rules.items():
                    # Check sequence distance constraint (linear positions)
                    if distance > rules['max_distance']:
                        continue

                    if self.biophysics.check_interaction(aa1, aa2, interaction_type):
                        found_interactions.add(interaction_type)

                if found_interactions:
                    # Update per-residue interaction profiles
                    residue_interactions[i].update(found_interactions)
                    residue_interactions[j].update(found_interactions)

                    # Select dominant interaction (highest strength coefficient)
                    dominant = max(found_interactions,
                                   key=lambda it: self.biophysics.interaction_rules[it]['strength'])

                    weight = self.biophysics.interaction_rules[dominant]['strength']

                    # Distance decay function (standard contact potential model)
                    # Linear decay reduces weight with increasing sequence separation
                    max_dist = self.biophysics.interaction_rules[dominant]['max_distance']
                    weight *= (1 - (distance / max_dist) * 0.3)

                    # HYBRID INTERACTION DETECTION
                    # Edge qualifies as hybrid if it satisfies both primary and secondary
                    # interaction types for any defined hybrid interaction
                    is_hybrid = 0
                    if len(found_interactions) >= 2:
                        for hybrid_type, rules in self.biophysics.hybrid_interactions.items():
                            if (rules['primary'] in found_interactions and
                                    rules['secondary'] in found_interactions):
                                is_hybrid = 1
                                hybrid_edge_counts[hybrid_type] += 1
                                weight *= rules['weight']  # Cooperative boost
                                break

                    edges.append((i, j))
                    edge_attributes["interaction_type"].append(dominant)
                    edge_attributes["weight"].append(min(weight, 1.0))  # Cap at 1.0
                    edge_attributes["interaction_set"].append(frozenset(found_interactions))
                    edge_attributes["is_hybrid"].append(is_hybrid)

        # Add all side-chain interaction edges to graph
        if edges:
            graph.add_edges(edges)
            for attr_name, attr_values in edge_attributes.items():
                graph.es[attr_name] = attr_values

        # ---------------------------------------------------------
        # 2. PEPTIDE BACKBONE (Sequence adjacency)
        #    Covalent bonds between consecutive residues
        # ---------------------------------------------------------
        backbone_edges = []
        for i in range(n - 1):
            if not graph.are_adjacent(i, i + 1):
                backbone_edges.append((i, i + 1))

        if backbone_edges:
            start_idx = graph.ecount()
            graph.add_edges(backbone_edges)

            # Set attributes for newly added backbone edges
            new_edge_slice = graph.es[start_idx:]
            new_edge_slice["weight"] = [1.0] * len(backbone_edges)
            new_edge_slice["interaction_type"] = ["backbone"] * len(backbone_edges)
            new_edge_slice["is_hybrid"] = [0] * len(backbone_edges)
            new_edge_slice["interaction_set"] = [frozenset(['backbone'])] * len(backbone_edges)

        # ---------------------------------------------------------
        # 3. HYBRID INTERACTION SCORE CALCULATION
        # ---------------------------------------------------------
        hybrid_scores = self._calculate_hybrid_scores(graph, residue_interactions, hybrid_edge_counts, seq)

        # Store hybrid scores as graph attributes
        for hybrid_type, score in hybrid_scores.items():
            graph[hybrid_type] = score

        return graph, hybrid_scores

    def _calculate_hybrid_scores(self, graph: ig.Graph, residue_interactions: Dict,
                                 hybrid_edge_counts: Dict[str, int], sequence: str) -> Dict[str, float]:
        """
        Compute normalized scores for each hybrid interaction type.

        Combines edge-based and residue-based evidence for hybrid interactions
        into composite scores. Edge-based scores reflect frequency in the graph,
        while residue-based scores reflect per-position interaction diversity.

        Args:
            graph: Constructed residue interaction graph
            residue_interactions: Dict mapping residue positions to set of
                                 interaction types they participate in
            hybrid_edge_counts: Count of edges qualifying as each hybrid type
            sequence: Original protein sequence (unused, preserved for API)

        Returns:
            Dict[str, float]: Normalized hybrid scores [0,1] for each hybrid type
                             Weighted combination: 0.6*edge_score + 0.4*residue_score
        """
        hybrid_scores = {hybrid_type: 0.0 for hybrid_type in self.biophysics.hybrid_interactions.keys()}

        if graph.ecount() == 0:
            return hybrid_scores

        # 1. Edge-based hybrid scores (normalized by total edges)
        total_edges = graph.ecount()
        for hybrid_type in self.biophysics.hybrid_interactions.keys():
            hybrid_scores[hybrid_type] = hybrid_edge_counts[hybrid_type] / total_edges if total_edges > 0 else 0

        # 2. Residue-based hybrid scores
        # Each residue with both primary and secondary interaction types contributes
        residue_hybrid_scores = {hybrid_type: [] for hybrid_type in self.biophysics.hybrid_interactions.keys()}

        for residue, interactions in residue_interactions.items():
            if len(interactions) >= 2:
                for hybrid_type, rules in self.biophysics.hybrid_interactions.items():
                    if rules['primary'] in interactions and rules['secondary'] in interactions:
                        # Unit contribution per qualifying residue
                        residue_hybrid_scores[hybrid_type].append(1.0)

        # 3. Combine edge and residue evidence
        for hybrid_type in self.biophysics.hybrid_interactions.keys():
            if residue_hybrid_scores[hybrid_type]:
                residue_mean = np.mean(residue_hybrid_scores[hybrid_type])
                hybrid_scores[hybrid_type] = 0.6 * hybrid_scores[hybrid_type] + 0.4 * residue_mean

        return hybrid_scores

    def extract_basic_graph_features(self, graph: ig.Graph) -> np.ndarray:
        """
        Extract comprehensive 85-dimensional graph-theoretic feature vector.

        Computes topological, statistical, and biologically-informed features
        from the residue interaction graph. Features span multiple categories:
        - Basic topology (node count, edge count, density)
        - Degree distributions (mean, variance, percentiles)
        - Weighted degree statistics
        - Interaction type composition (13 types + backbone)
        - Centrality measures (betweenness, closeness, eigenvector)
        - Community structure (modularity, component analysis)
        - Localization-specific patterns (N-term, C-term, charge clusters)
        - Path-based metrics (efficiency, diameter, length distribution)
        - Edge weight distributions

        Returns:
            np.ndarray: 85-element float32 feature vector normalized where applicable
                        Zero-padded for missing values or insufficient graph size

        Notes:
            - Safe fallback to zero vectors for any failed computation
            - All distance metrics are graph-theoretic, not spatial
            - Centrality warnings filtered to prevent console pollution
        """
        features = []

        n = graph.vcount()
        e = graph.ecount()

        # 1. Basic topology
        features.extend([n, e, e / max(1, (n * (n - 1) / 2)) if n > 1 else 0])

        # 2. Degree statistics
        if e > 0:
            degrees = graph.degree()
            features.extend([
                np.mean(degrees), np.std(degrees), np.max(degrees),
                np.percentile(degrees, 25), np.median(degrees), np.percentile(degrees, 75)
            ])
        else:
            features.extend([0.0] * 6)

        # 3. Weighted degree (strength)
        if e > 0 and 'weight' in graph.es.attributes():
            weights = graph.es["weight"]
            weighted_degrees = graph.strength(weights=weights)
            features.extend([
                np.mean(weighted_degrees), np.std(weighted_degrees), np.max(weighted_degrees)
            ])
        else:
            features.extend([0.0] * 3)

        # 4. Interaction type distribution
        # Includes 13 biophysical interaction types + backbone (14 total)
        if e > 0 and 'interaction_type' in graph.es.attributes():
            interaction_counts = Counter(graph.es["interaction_type"])

            # Counts for standard biochemical interactions (excluding backbone)
            for interaction_type in self.biophysics.interaction_rules.keys():
                if interaction_type == 'backbone':
                    continue
                features.append(interaction_counts.get(interaction_type, 0) / e)

            # Backbone edge ratio
            features.append(interaction_counts.get('backbone', 0) / e)

        else:
            # Matches the e>0 branch exactly: one ratio per interaction type
            # (backbone included), not len(...)+1 - that extra zero was
            # silently shifting every feature after this block by one column
            # for edgeless (single-residue) graphs.
            features.extend([0.0] * len(self.biophysics.interaction_rules))

        # 5. Centrality features
        centrality_features = self._extract_centrality(graph)
        features.extend(centrality_features)

        # 6. Community features
        community_features = self._extract_community_features(graph)
        features.extend(community_features)

        # 7. Localization pattern features
        sequence = ''.join(graph.vs["residue"]) if "residue" in graph.vs.attributes() else ""
        pattern_features = self._extract_localization_patterns(graph, sequence)
        features.extend(pattern_features)

        # 8. Path features
        path_features = self._extract_path_features(graph)
        features.extend(path_features)

        # 9. Additional metrics
        additional_features = self._extract_additional_metrics(graph)
        features.extend(additional_features)

        # Ensure consistent feature dimension (85 features)
        target_len = 85
        if len(features) < target_len:
            features.extend([0.0] * (target_len - len(features)))

        return np.array(features, dtype=np.float32)

    def _extract_centrality(self, graph: ig.Graph) -> List[float]:
        """
        Extract node centrality measures as graph-level summary statistics.

        Computes betweenness, closeness, and eigenvector centrality distributions,
        reducing to mean, standard deviation, and range statistics.

        Args:
            graph: Residue interaction graph

        Returns:
            List[float]: 12-element feature vector
                [betweenness_mean, betweenness_std, betweenness_max, betweenness_min,
                 betweenness_25p, betweenness_75p, closeness_mean, closeness_std,
                 closeness_max, closeness_min, eigenvector_mean, eigenvector_std]
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if e == 0 or n < 3:
            return [0.0] * 12

        try:
            weights = graph.es["weight"] if "weight" in graph.es.attributes() else None

            # Betweenness centrality
            betweenness = graph.betweenness(weights=weights)
            features.extend([
                np.mean(betweenness), np.std(betweenness),
                np.max(betweenness), np.min(betweenness),
                np.percentile(betweenness, 25), np.percentile(betweenness, 75)
            ])

            # Closeness centrality
            closeness = graph.closeness(weights=weights)
            features.extend([
                np.mean(closeness), np.std(closeness),
                np.max(closeness), np.min(closeness)
            ])

            # Eigenvector centrality
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*eigenvector centralities are nearly zero.*"
                    )
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    eigenvector = graph.eigenvector_centrality(weights=weights)
                    features.extend([np.mean(eigenvector), np.std(eigenvector)])
            except Exception:
                features.extend([0.0, 0.0])

        except Exception:
            return [0.0] * 12

        return features

    def _extract_community_features(self, graph: ig.Graph) -> List[float]:
        """
        Extract community structure features using multilevel modularity optimization.

        Identifies densely connected subgraphs (putative structural/functional domains)
        and computes metrics describing their organization.

        Args:
            graph: Residue interaction graph

        Returns:
            List[float]: 8-element feature vector
                [n_communities, modularity_score, mean_community_size, std_community_size,
                 max_community_size, min_community_size, intra_edge_ratio, inter_edge_ratio]
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if e < 10 or n < 20:
            return [0.0] * 8

        try:
            weights = graph.es["weight"] if 'weight' in graph.es.attributes() else None
            communities = graph.community_multilevel(weights=weights)

            community_sizes = [len(c) for c in communities]
            features.extend([
                len(communities),
                graph.modularity(communities, weights=weights),
                np.mean(community_sizes) if community_sizes else 0,
                np.std(community_sizes) if len(community_sizes) > 1 else 0,
                np.max(community_sizes) if community_sizes else 0,
                np.min(community_sizes) if community_sizes else 0,
            ])

            # Intra- vs inter-community edge distribution
            if len(communities) > 1:
                community_map = {}
                for comm_id, community in enumerate(communities):
                    for node in community:
                        community_map[node] = comm_id

                intra_edges = 0
                inter_edges = 0
                for edge in graph.es:
                    if community_map[edge.source] == community_map[edge.target]:
                        intra_edges += 1
                    else:
                        inter_edges += 1

                total = intra_edges + inter_edges
                features.extend([
                    intra_edges / total if total > 0 else 0,
                    inter_edges / total if total > 0 else 0
                ])
            else:
                features.extend([1.0, 0.0])

        except Exception:
            features.extend([0.0] * 8)

        return features

    def _extract_localization_patterns(self, graph: ig.Graph, sequence: str) -> np.ndarray:
        """
        Extract graph features specific to subcellular localization signals.

        Analyzes graph topology in functionally annotated regions:
        - N-terminal signal peptides and targeting sequences
        - C-terminal retention and retrieval motifs
        - Basic residue clusters (nuclear localization signals)
        - Hydrophobic clusters (transmembrane domains)
        - Charge distribution patterns

        Args:
            graph: Residue interaction graph
            sequence: Amino acid sequence for residue type identification

        Returns:
            np.ndarray: 20-element feature vector encoding localization-relevant
                       graph properties
        """
        features = []

        n = graph.vcount()
        if n < 10:
            return np.zeros(20, dtype=np.float32)

        # 1. N-terminal region analysis (positions 1-30)
        n_term_features = self._analyze_n_terminal_graph(graph, sequence)
        features.extend(n_term_features)

        # 2. C-terminal region analysis (last 10 residues)
        c_term_features = self._analyze_c_terminal_graph(graph, sequence)
        features.extend(c_term_features)

        # 3. Basic residue cluster detection (NLS signals)
        basic_features = self._detect_basic_clusters_graph(graph, sequence)
        features.extend(basic_features)

        # 4. Hydrophobic cluster analysis (TM domains)
        hydrophobic_features = self._detect_hydrophobic_clusters(graph, sequence)
        features.extend(hydrophobic_features)

        # 5. Charge distribution patterns
        charge_features = self._analyze_charge_distribution(graph, sequence)
        features.extend(charge_features)

        return np.array(features, dtype=np.float32)

    def _analyze_n_terminal_graph(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze N-terminal subgraph properties relevant to signal peptides.

        Computes metrics for residues 1-30, focusing on features predictive
        of signal peptide function and targeting efficiency.

        Args:
            graph: Complete residue interaction graph
            sequence: Full amino acid sequence

        Returns:
            List[float]: 4-element feature vector
                - Hydrophobic density in first half (h-region)
                - Basic residue centrality in n-region
                - Local clustering coefficient (flexibility proxy)
                - Edge density (structural compactness)
        """
        features = []

        n_region = min(30, len(sequence))
        if n_region < 10:
            return [0.0] * 4

        try:
            n_nodes = list(range(n_region))
            n_subgraph = graph.induced_subgraph(n_nodes)

            if n_subgraph.ecount() > 0:
                # 1. Hydrophobic density in h-region (signal peptide core)
                first_half = n_nodes[:n_region // 2]
                hydrophobic_first = sum(1 for i in first_half if sequence[i] in 'AVILMFYW')
                features.append(hydrophobic_first / len(first_half))

                # 2. Basic residue centrality in n-region (NLS signals)
                basic_nodes = [i for i in n_nodes if sequence[i] in 'KRH']
                if basic_nodes:
                    betweenness = n_subgraph.betweenness()
                    basic_centrality = np.mean([betweenness[i] for i in basic_nodes if i < len(betweenness)])
                    features.append(basic_centrality)
                else:
                    features.append(0.0)

                # 3. Average local clustering coefficient (conformational flexibility)
                clustering = n_subgraph.transitivity_avglocal_undirected()
                features.append(clustering)

                # 4. Edge density (structural compactness)
                max_edges = (n_region * (n_region - 1)) / 2
                actual_edges = n_subgraph.ecount()
                features.append(actual_edges / max_edges if max_edges > 0 else 0)
            else:
                features.extend([0.0] * 4)

        except Exception:
            features.extend([0.0] * 4)

        return features

    def _analyze_c_terminal_graph(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze C-terminal subgraph properties relevant to retention signals.

        Computes metrics for last 10 residues, focusing on features relevant
        to ER retention, endocytosis, and sorting signals.

        Args:
            graph: Complete residue interaction graph
            sequence: Full amino acid sequence

        Returns:
            List[float]: 3-element feature vector
                - Mean degree centrality (surface exposure proxy)
                - Charged residue connectivity
                - Terminal residue connectivity
        """
        features = []

        c_region = min(10, len(sequence))
        if c_region < 3:
            return [0.0] * 3

        try:
            c_start = len(sequence) - c_region
            c_nodes = list(range(c_start, len(sequence)))
            c_subgraph = graph.induced_subgraph(c_nodes)

            if c_subgraph.ecount() > 0:
                # 1. Surface exposure proxy (mean degree centrality)
                degrees = c_subgraph.degree()
                features.append(np.mean(degrees) if degrees else 0)

                # 2. Charged residue connectivity in C-terminus
                charged_nodes = [i - c_start for i in c_nodes if sequence[i] in 'KRHDE']
                if charged_nodes:
                    charged_edges = 0
                    for i in range(len(charged_nodes)):
                        for j in range(i + 1, len(charged_nodes)):
                            if c_subgraph.are_adjacent(charged_nodes[i], charged_nodes[j]):
                                charged_edges += 1
                    features.append(charged_edges / max(1, len(charged_nodes)))
                else:
                    features.append(0.0)

                # 3. Terminal residue connectivity to protein body
                last_node = len(sequence) - 1
                if last_node < graph.vcount():
                    terminal_degree = graph.degree(last_node)
                    features.append(terminal_degree / 10.0)  # Normalized
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 3)

        except Exception:
            features.extend([0.0] * 3)

        return features

    def _detect_basic_clusters_graph(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Detect and characterize basic residue clusters (putative NLS signals).

        Analyzes connectivity patterns among lysine, arginine, and histidine
        residues to identify nuclear localization signal-like clusters.

        Args:
            graph: Complete residue interaction graph
            sequence: Full amino acid sequence

        Returns:
            List[float]: 4-element feature vector
                - Basic residue connectivity density
                - Mean betweenness centrality of basic residues
                - Number of basic clusters (connected components)
                - Size of largest basic cluster
        """
        features = []

        # Identify all basic residue positions
        basic_positions = [i for i, aa in enumerate(sequence) if aa in 'KRH']

        if len(basic_positions) < 2:
            return [0.0] * 4

        # 1. Basic residue connectivity density
        basic_edges = 0
        for i in range(len(basic_positions)):
            for j in range(i + 1, len(basic_positions)):
                if graph.are_adjacent(basic_positions[i], basic_positions[j]):
                    basic_edges += 1

        max_possible = (len(basic_positions) * (len(basic_positions) - 1)) / 2
        basic_connectivity = basic_edges / max_possible if max_possible > 0 else 0
        features.append(basic_connectivity)

        # 2. Basic residue centrality (structural prominence)
        if basic_positions and graph.ecount() > 0:
            betweenness = graph.betweenness()
            basic_centrality = np.mean([betweenness[pos] for pos in basic_positions if pos < len(betweenness)])
            features.append(basic_centrality)
        else:
            features.append(0.0)

        # 3. Cluster size distribution
        # Extract subgraph of basic residues and identify connected components
        basic_subgraph = graph.induced_subgraph(basic_positions)
        basic_components = basic_subgraph.connected_components()
        component_sizes = [len(c) for c in basic_components]

        if component_sizes:
            features.extend([
                len(component_sizes),  # Number of distinct basic clusters
                np.max(component_sizes)  # Largest cluster size
            ])
        else:
            features.extend([0.0, 0.0])

        return features

    def _detect_hydrophobic_clusters(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Detect and characterize hydrophobic clusters (putative TM domains).

        Analyzes connectivity patterns among hydrophobic residues to identify
        potential transmembrane helices and signal peptide h-regions.

        Args:
            graph: Complete residue interaction graph
            sequence: Full amino acid sequence

        Returns:
            List[float]: 5-element feature vector
                - Hydrophobic connectivity density
                - External connectivity ratio (buried vs exposed)
                - Number of hydrophobic clusters
                - Largest cluster size
                - Mean cluster size
        """
        features = []

        # Identify hydrophobic residue positions
        hydrophobic_positions = [i for i, aa in enumerate(sequence) if aa in 'AVILMFYW']

        if len(hydrophobic_positions) < 5:
            return [0.0] * 5

        # Create hydrophobic residue subgraph
        hydrophobic_subgraph = graph.induced_subgraph(hydrophobic_positions)

        if hydrophobic_subgraph.ecount() == 0:
            return [0.0] * 5

        # 1. Hydrophobic cluster internal connectivity
        max_h_edges = (len(hydrophobic_positions) * (len(hydrophobic_positions) - 1)) / 2
        actual_h_edges = hydrophobic_subgraph.ecount()
        features.append(actual_h_edges / max_h_edges if max_h_edges > 0 else 0)

        # 2. External connectivity (buriedness proxy)
        external_edges = 0
        for pos in hydrophobic_positions:
            neighbors = graph.neighbors(pos)
            for neighbor in neighbors:
                if sequence[neighbor] not in 'AVILMFYW':
                    external_edges += 1

        features.append(external_edges / len(hydrophobic_positions))

        # 3. Cluster component analysis
        h_components = hydrophobic_subgraph.connected_components()
        h_component_sizes = [len(c) for c in h_components]

        if h_component_sizes:
            features.extend([
                len(h_components),  # Number of hydrophobic clusters
                np.max(h_component_sizes),  # Largest hydrophobic cluster
                np.mean(h_component_sizes)  # Average cluster size
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _analyze_charge_distribution(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze charge distribution patterns across sequence thirds.

        Computes regional densities of basic (KRH) and acidic (DE) residues
        to identify charge gradients relevant to subcellular localization.

        Args:
            graph: Complete residue interaction graph (unused, preserved for API)
            sequence: Full amino acid sequence

        Returns:
            List[float]: 4-element feature vector
                - N-terminal basic density
                - Basic charge gradient (N-term - C-term)
                - Acidic charge gradient (C-term - N-term)
                - Normalized net charge
        """
        features = []

        n = len(sequence)
        if n < 10:
            return [0.0] * 4

        # Partition sequence into thirds
        third = n // 3
        regions = [
            list(range(0, third)),
            list(range(third, 2 * third)),
            list(range(2 * third, n))
        ]

        try:
            # Calculate basic/acidic residue density in each region
            basic_densities = []
            acidic_densities = []

            for region in regions:
                if region:
                    region_basic = sum(1 for i in region if sequence[i] in 'KRH')
                    region_acidic = sum(1 for i in region if sequence[i] in 'DE')
                    basic_densities.append(region_basic / len(region))
                    acidic_densities.append(region_acidic / len(region))

            # 1. N-terminal basic density (signal peptide/MTS feature)
            if len(basic_densities) > 0:
                features.append(basic_densities[0])
            else:
                features.append(0.0)

            # 2. Charge gradients
            if len(basic_densities) > 2 and len(acidic_densities) > 2:
                basic_gradient = basic_densities[0] - basic_densities[2]
                acidic_gradient = acidic_densities[2] - acidic_densities[0]
                features.extend([basic_gradient, acidic_gradient])
            else:
                features.extend([0.0, 0.0])

            # 3. Overall charge balance
            total_basic = sum(1 for aa in sequence if aa in 'KRH')
            total_acidic = sum(1 for aa in sequence if aa in 'DE')
            features.append((total_basic - total_acidic) / n if n > 0 else 0)

        except Exception:
            features.extend([0.0] * 4)

        return features

    def _extract_path_features(self, graph: ig.Graph) -> List[float]:
        """
        Extract path-based topological features using efficient sampling.

        Computes metrics related to shortest path distributions and network
        efficiency. For disconnected graphs, operates on largest connected component.

        Args:
            graph: Residue interaction graph

        Returns:
            List[float]: 6-element feature vector
                - Average shortest path length (LCC)
                - Graph diameter (LCC)
                - Global efficiency (1/avg_path)
                - Average local efficiency (sampled)
                - Path length standard deviation (sampled)
                - Maximum path length (sampled)
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if n < 3 or e == 0:
            return [0.0] * 6

        try:
            weights = graph.es["weight"] if 'weight' in graph.es.attributes() else None

            # ==========================================================
            # 1. LCC Analysis (Avg Path, Diameter, Global Efficiency)
            # ==========================================================
            # Use largest connected component to avoid infinite distances
            if not graph.is_connected():
                clusters = graph.connected_components()
                subgraph = clusters.giant()
                sub_weights = subgraph.es["weight"] if 'weight' in subgraph.es.attributes() else None
            else:
                subgraph = graph
                sub_weights = weights

            # Compute standard metrics on connected component
            avg_path = subgraph.average_path_length(weights=sub_weights, directed=False)
            diameter = subgraph.diameter(weights=sub_weights, directed=False)
            global_efficiency = 1.0 / avg_path if avg_path > 0 else 0.0

            features.extend([avg_path, diameter, global_efficiency])

            # ==========================================================
            # 2. True Local Efficiency (Latora-Marchiori definition)
            #    Computed via sampling for computational efficiency
            # ==========================================================
            sample_size = min(30, n)
            sample_nodes = random.sample(range(n), sample_size)
            local_effs = []

            for v in sample_nodes:
                neighbors = graph.neighbors(v)
                if len(neighbors) < 2:
                    local_effs.append(0.0)
                    continue

                # Induced subgraph of neighbors (excluding vertex v)
                neigh_graph = graph.induced_subgraph(neighbors)
                if neigh_graph.ecount() == 0:
                    local_effs.append(0.0)
                    continue

                # Calculate efficiency of neighbor subgraph
                try:
                    dists = neigh_graph.distances(weights=neigh_graph.es["weight"] if weights else None)

                    inv_dist_sum = 0.0
                    k = len(neighbors)

                    # Sum inverse distances for upper triangle
                    for r in range(k):
                        for c in range(r + 1, k):
                            d = dists[r][c]
                            if d > 0 and not np.isinf(d):
                                inv_dist_sum += (1.0 / d)

                    # Normalize: 2 * sum / (k * (k-1))
                    if k > 1:
                        eff = (2.0 * inv_dist_sum) / (k * (k - 1))
                        local_effs.append(eff)
                    else:
                        local_effs.append(0.0)
                except Exception:
                    local_effs.append(0.0)

            avg_local_efficiency = np.mean(local_effs) if local_effs else 0.0
            features.append(avg_local_efficiency)

            # ==========================================================
            # 3. Path Length Statistics (Std & Max) - Sampled
            # ==========================================================
            try:
                n_samples = min(50, n)
                sample_indices = random.sample(range(n), n_samples)

                # Calculate paths from sample nodes to all others
                sampled_paths = graph.distances(source=sample_indices, weights=weights)

                # Collect finite, positive distances
                all_sampled_distances = []
                for row in sampled_paths:
                    for d in row:
                        if d > 0 and not np.isinf(d):
                            all_sampled_distances.append(d)

                if all_sampled_distances:
                    path_std = np.std(all_sampled_distances)
                    max_path = np.max(all_sampled_distances)
                else:
                    path_std = 0.0
                    max_path = 0.0
            except Exception:
                path_std = 0.0
                max_path = 0.0

            features.extend([path_std, max_path])

        except Exception:
            # Fallback for any numerical errors
            current_len = len(features)
            features.extend([0.0] * (6 - current_len))

        return features

    def _extract_additional_metrics(self, graph: ig.Graph) -> List[float]:
        """
        Extract supplementary graph metrics for comprehensive feature coverage.

        Computes clustering coefficients, assortativity, regional densities,
        and edge weight distribution statistics.

        Args:
            graph: Residue interaction graph

        Returns:
            List[float]: 15-element feature vector
                - Average local clustering coefficient
                - Degree assortativity
                - Graph radius (if connected)
                - Triangle count
                - N-terminal density (positions 1-30)
                - Middle region density
                - C-terminal density (last 30 positions)
                - Edge weight statistics (8 features)
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if e == 0 or n < 3:
            return [0.0] * 15

        try:
            # 1. Average local clustering coefficient
            if e > 0:
                weights = graph.es["weight"] if 'weight' in graph.es.attributes() else None
                clustering = graph.transitivity_avglocal_undirected(weights=weights)
                features.append(clustering)
            else:
                features.append(0.0)

            # 2. Degree assortativity
            try:
                assortativity = graph.assortativity_degree()
                features.append(assortativity)
            except Exception:
                features.append(0.0)

            # 3. Graph radius (if connected)
            try:
                if graph.is_connected():
                    radius = graph.radius()
                    features.append(radius)
                else:
                    features.append(0.0)
            except Exception:
                features.append(0.0)

            # 4. Triangle count (3-cliques)
            triangles = graph.cliques(min=3, max=3)
            features.append(len(triangles))

            # 5. N-terminal region density (positions 1-30)
            n_region = min(30, n)
            if n_region > 0:
                n_edges = len([e for e in graph.es if e.source < n_region and e.target < n_region])
                max_possible = (n_region * (n_region - 1)) / 2
                features.append(n_edges / max_possible if max_possible > 0 else 0)
            else:
                features.append(0.0)

            # 6. Middle region density (positions 30 to n-30)
            if n > 60:
                mid_start = 30
                mid_end = n - 30
                mid_edges = len([e for e in graph.es
                                 if mid_start <= e.source < mid_end
                                 and mid_start <= e.target < mid_end])
                mid_max = ((mid_end - mid_start) * (mid_end - mid_start - 1)) / 2
                features.append(mid_edges / mid_max if mid_max > 0 else 0)
            else:
                features.append(0.0)

            # 7. C-terminal region density (last 30 positions)
            c_region = min(30, n)
            if c_region > 0:
                c_start = n - c_region
                c_edges = len([e for e in graph.es
                               if e.source >= c_start and e.target >= c_start])
                c_max = (c_region * (c_region - 1)) / 2
                features.append(c_edges / c_max if c_max > 0 else 0)
            else:
                features.append(0.0)

            # 8. Edge weight distribution statistics
            if e > 0 and 'weight' in graph.es.attributes():
                weights = graph.es["weight"]
                features.extend([
                    np.mean(weights), np.std(weights),
                    np.max(weights), np.min(weights),
                    np.percentile(weights, 25), np.median(weights),
                    np.percentile(weights, 75),
                    np.max(weights) - np.min(weights)  # Range
                ])
            else:
                features.extend([0.0] * 8)

        except Exception:
            features.extend([0.0] * 15)

        return features

    def extract_hybrid_features(self, graph: ig.Graph, hybrid_scores: Dict[str, float], sequence: str) -> np.ndarray:
        """
        Extract 22-dimensional feature vector specific to hybrid interactions.

        Computes features describing the prevalence, distribution, and organization
        of hybrid interactions (edges satisfying multiple interaction types).

        Args:
            graph: Residue interaction graph
            hybrid_scores: Dictionary of hybrid interaction scores from _calculate_hybrid_scores
            sequence: Amino acid sequence for regional analysis

        Returns:
            np.ndarray: 22-element float32 feature vector encoding:
                - 9 raw hybrid scores (each hybrid type)
                - 4 regional hybrid densities (N-term, C-term, hydrophobic, basic)
                - 4 hybrid network properties (ratio, density, path length, components)
                - 3 top hybrid co-occurrence scores
                - 3 advanced metrics (diversity, centrality, clustering)
        """
        features = []

        if graph.ecount() == 0:
            return np.zeros(22, dtype=np.float32)

        # 1. Basic hybrid scores (9 features)
        for hybrid_type in self.biophysics.hybrid_interactions.keys():
            features.append(hybrid_scores.get(hybrid_type, 0.0))

        # 2. Regional hybrid density (4 features)
        seq = sequence.upper()
        regions = {
            'n_terminal': list(range(0, min(30, len(seq)))),
            'c_terminal': list(range(max(0, len(seq) - 10), len(seq))),
            'hydrophobic_regions': [i for i, aa in enumerate(seq) if aa in 'AVILMFYW'],
            'basic_regions': [i for i, aa in enumerate(seq) if aa in 'KRH']
        }

        for region_name, region_nodes in regions.items():
            if len(region_nodes) < 3:
                features.append(0.0)
                continue

            # Count hybrid edges within region
            hybrid_edges = 0
            total_region_edges = 0

            for i in range(len(region_nodes)):
                for j in range(i + 1, len(region_nodes)):
                    node_i = region_nodes[i]
                    node_j = region_nodes[j]

                    if graph.are_adjacent(node_i, node_j):
                        total_region_edges += 1
                        edge_id = graph.get_eid(node_i, node_j)
                        if graph.es[edge_id]["is_hybrid"] == 1:
                            hybrid_edges += 1

            if total_region_edges > 0:
                features.append(hybrid_edges / total_region_edges)
            else:
                features.append(0.0)

        # 3. Hybrid network properties (4 features)
        try:
            hybrid_edge_indices = [i for i, edge in enumerate(graph.es) if edge["is_hybrid"] == 1]

            if hybrid_edge_indices:
                hybrid_subgraph = graph.subgraph_edges(hybrid_edge_indices, delete_vertices=False)

                features.extend([
                    len(hybrid_edge_indices) / graph.ecount(),  # Hybrid edge ratio
                    hybrid_subgraph.density(),
                    hybrid_subgraph.average_path_length() if hybrid_subgraph.vcount() > 1 else 0,
                    len(hybrid_subgraph.connected_components())  # Hybrid cluster count
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 4. Hybrid co-occurrence patterns (3 features)
        cooccurrence_scores = self._calculate_hybrid_cooccurrence(graph)
        top_scores = sorted(cooccurrence_scores.values(), reverse=True)[:3]
        features.extend(top_scores + [0.0] * (3 - len(top_scores)))

        # 5. Advanced hybrid metrics (3 features)
        advanced_metrics = self._calculate_advanced_hybrid_metrics(graph, hybrid_scores)
        features.extend(advanced_metrics)

        return np.array(features, dtype=np.float32)

    def _calculate_hybrid_cooccurrence(self, graph: ig.Graph) -> Dict[Tuple[str, str], float]:
        """
        Calculate co-occurrence frequencies of different hybrid types.

        For edges that qualify as multiple hybrid types simultaneously, counts
        pairwise co-occurrences and normalizes by total hybrid edges.

        Args:
            graph: Residue interaction graph

        Returns:
            Dict[Tuple[str, str], float]: Normalized co-occurrence frequency
                                         for each unordered hybrid pair
        """
        cooccurrence = {}
        hybrid_types = list(self.biophysics.hybrid_interactions.keys())

        # Initialize co-occurrence pairs
        for i in range(len(hybrid_types)):
            for j in range(i + 1, len(hybrid_types)):
                pair = (hybrid_types[i], hybrid_types[j])
                cooccurrence[pair] = 0

        # Count co-occurrences across edges
        for edge in graph.es:
            if edge["is_hybrid"] == 1:
                interaction_set = edge["interaction_set"]

                # Identify all hybrid types this edge satisfies
                edge_hybrids = set()
                for hybrid_type, rules in self.biophysics.hybrid_interactions.items():
                    if rules['primary'] in interaction_set and rules['secondary'] in interaction_set:
                        edge_hybrids.add(hybrid_type)

                # Update pairwise co-occurrence counts
                for hybrid1 in edge_hybrids:
                    for hybrid2 in edge_hybrids:
                        if hybrid1 != hybrid2:
                            pair = tuple(sorted([hybrid1, hybrid2]))
                            cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

        # Normalize by total hybrid edges
        total_hybrid_edges = sum(1 for edge in graph.es if edge["is_hybrid"] == 1)
        if total_hybrid_edges > 0:
            for key in cooccurrence:
                cooccurrence[key] /= total_hybrid_edges

        return cooccurrence

    def _calculate_advanced_hybrid_metrics(self, graph: ig.Graph, hybrid_scores: Dict[str, float]) -> List[float]:
        """
        Compute advanced organizational metrics for hybrid interaction network.

        Args:
            graph: Residue interaction graph
            hybrid_scores: Dictionary of hybrid interaction scores

        Returns:
            List[float]: 3-element feature vector
                - Normalized hybrid diversity (Shannon entropy)
                - Mean betweenness centrality of hybrid subgraph
                - Average local clustering coefficient of hybrid subgraph
        """
        metrics = []

        # 1. Hybrid diversity (Shannon entropy)
        hybrid_values = np.array(list(hybrid_scores.values()))
        if np.sum(hybrid_values) > 0:
            normalized = hybrid_values / np.sum(hybrid_values)
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            metrics.append(entropy / np.log(len(hybrid_values)))  # Normalized [0,1]
        else:
            metrics.append(0.0)

        # 2. Hybrid network centrality
        try:
            hybrid_edges = [i for i, edge in enumerate(graph.es) if edge["is_hybrid"] == 1]
            if hybrid_edges:
                hybrid_subgraph = graph.subgraph_edges(hybrid_edges, delete_vertices=False)
                if hybrid_subgraph.ecount() > 0:
                    betweenness = hybrid_subgraph.betweenness()
                    metrics.append(np.mean(betweenness) if betweenness else 0)
                else:
                    metrics.append(0.0)
            else:
                metrics.append(0.0)
        except Exception:
            metrics.append(0.0)

        # 3. Hybrid network clustering
        try:
            hybrid_edges = [i for i, edge in enumerate(graph.es) if edge["is_hybrid"] == 1]
            if hybrid_edges:
                hybrid_subgraph = graph.subgraph_edges(hybrid_edges, delete_vertices=False)
                if hybrid_subgraph.vcount() > 2 and hybrid_subgraph.ecount() > 0:
                    clustering = hybrid_subgraph.transitivity_avglocal_undirected()
                    metrics.append(clustering)
                else:
                    metrics.append(0.0)
            else:
                metrics.append(0.0)
        except Exception:
            metrics.append(0.0)

        return metrics