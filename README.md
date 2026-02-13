# BioGraphX
[![DOI](https://img.shields.io/badge/DOI-10.64898%252F2026.01.21.700873-blue)](https://doi.org/10.64898/2026.01.21.700873)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-blue.svg)

**BioGraphX** is a physicochemical graph encoding framework that transforms protein primary sequences into high-fidelity 158-dimensional biophysical feature vectors. It models proteins as residue interaction networks based on fundamental biochemical principles, enabling structure-independent sequence analysis for downstream machine learning applications.
---

## 🎯 Key Features

### 🧬Architecture
* **Biophysics Engine**: Models 13 non-covalent interaction types (hydrophobic, H-bond, salt bridges, disulfide, π-interactions, cation-π, van der Waals, CH-π, NH-π, carbonyl-carbonyl, sulfur-π, and peptide backbone) with linear sequence distance constraints.
* **Adaptive Processor**: Smart truncation (40% N-term / 20% Core / 40% C-term) preserving signal peptides, transmembrane domains, and retention signals..
* **Graph Construction Engine**: Residue-level interaction networks with hybrid interaction tracking and cooperative binding scores.
* **Frustration Analysis**: Per-residue conformational conflict detection from constraint graph topology. 
* **Localization Profiler**: Scans for compartment-specific motifs (e.g., AREs, Shine-Dalgarno, Nuclear clusters).

### 📊 Comprehensive Feature Extraction
| Feature Category | Count | Description |
| :--- | :---: | :--- |
| **Graph Topology** | 85 | Degree distributions, centrality measures (betweenness/closeness/eigenvector), community structure, path efficiency, clustering coefficients, regional densities |
| **Hybrid Features** | 23 | Cooperative interaction scores, regional hybrid enrichment, co-occurrence patterns, network diversity metrics |
| **Knowledge Profiles** | 20 | N-terminal signal features, C-terminal retention signals, basic cluster (NLS) detection, hydrophobic cluster (TM) prediction, charge gradients |
| **Frustration Analysis** | 11 | Per-residue frustration, signal vs structural contrast, hotspot detection, satisfaction ratios, profile correlation |
| **Global Physics** | 19 | Isoelectric point (pI), GRAVY hydrophobicity, net charge, aromaticity, instability proxy, autocorrelation (hydrophobicity/charge lags 1-6), Shannon entropy |

---

## 🏗️ Architecture
```text
BioGraphX Pipeline
├── BioPhysicsStrategy (13 Interaction Rules + Hybrid Detection)
├── SequencePreprocessor (Motif-Preserving Truncation & Windowing)
├── MotifProfiler (Compartment-specific Localization Scoring)
├── GraphEngine (Residue Interaction Network Construction)
└── FrustrationAnalyzer (Conformational Conflict Detection)

```
---

## Quick Start
### Installation
# Clone repository
```text
git clone https://github.com/Abubakar-Saeed/BioGraphX.git
cd BioGraphX


```
# Install dependencies
pip install numpy pandas scipy igraph torch torchvision torchaudio scikit-learn tqdm

### Basic Usage
```text

from src.biographx.pipeline import BioGraphXPipeline

# Initialize pipeline
pipeline = BioGraphXPipeline()

# Process a single protein sequence
sequence = "MKTIIALSYIFCLVFADYKDDDDK"
features = pipeline.process_sequence(sequence)

print(f"Extracted {len(features)} biophysical features")
print(f"GRAVY hydrophobicity: {features['gravy']:.3f}")
print(f"Isoelectric point: {features['isoelectric_point']:.2f}")
print(f"N-terminal frustration: {features['Frustration_NTerminal_Mean']:.3f}")
```

### Process a single protein sequence
```text
sequence = "MKTIIALSYIFCLVFADYKDDDDK"
features = pipeline.process_sequence(sequence)

print(f"Extracted {len(features)} features")
```
### Batch Processing from CSV
```text
from biographx.pipeline import run_integrated_pipeline

# Process large-scale proteomics datasets
run_integrated_pipeline(
    input_file="proteins.csv",      # Requires 'ACC' and 'Sequence' columns
    output_file="encoded_features.csv",
    chunk_size=500,                # Sequences per chunk
    n_jobs=10                     # Parallel workers
)
```
## Scientific Basis
BioGraphX represents proteins as mathematical graphs where nodes are amino acid residues and edges represent potential non-covalent interactions. All distance constraints are measured in linear sequence positions (amino acid indices), NOT 3D Ångström spatial coordinates, enabling structure-independent analysis.
### Localization Patterns
The pipeline identifies specific interaction patterns associated with:
* **Nuclear:** Basic clusters (NLS), bipartite moti.
* **Mitochondrion:** N-terminal amphipathic helices, MTS.
* **Extracellular:** Signal peptides, disulfide-rich.
* **ER/Golgi:** KDEL, KKXX, dileucine motifs.
## Citation
If you use BioGraphX in your research, please cite:
Saeed, A., & Abbas, W. (2026). BioGraphX: Bridging the sequence–structure gap via physicochemical graph encoding for explainable subcellular localization prediction. https://doi.org/10.64898/2026.01.21.700873
