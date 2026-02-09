# Interpretable_VAE_project for Transcription Factor Rewiring

Overview:
This project implements an interpretable Variational Autoencoder (VAE) to study transcription factor (TF) rewiring in the context of the interferon response. The focus is on interferon-related TFs such as STAT1, STAT2, and IRFs, and how their regulatory influence on target genes differs between stimulated and control conditions.

The model is based on the VEGA (Variational Autoencoder enhanced by Gene Annotations) framework, which incorporates prior biological knowledge directly into the decoder to ensure interpretability of latent variables.

# Data
1) Gene expression data from PBMCs under control and interferon-stimulated conditions
2) CollecTRI database used to define transcription factor–gene regulatory interactions
3) TF–gene annotations are encoded as a decoder mask, constraining the model architecture

## Model Architecture

### VEGA (Interpretable Variational Autoencoder)

#### Encoder
- Two fully connected layers with dropout
- Outputs latent mean and variance
- Uses the reparameterization trick to enable backpropagation through stochastic latent variables

#### Decoder
- Linear decoder constrained by a gene regulatory mask
- Mask enforces biologically meaningful connections between latent transcription factor (TF) nodes and genes
- Latent dimensions correspond directly to transcription factors, enabling interpretability

---

## Training Strategy

- Separate models are trained for:
  - **Control** condition
  - **Stimulated** condition
- Models are trained multiple times to account for stochasticity inherent to VAEs
- Variability in decoder weights across runs is used to estimate uncertainty in TF–gene associations

---

## Uncertainty Estimation Approaches

Three complementary approaches were explored:

### 1. Basic VEGA (Ensemble Training)
- Train multiple independent VEGA models
- Compute mean and standard deviation of decoder weights
- Identify TF–gene interactions with high or low certainty

### 2. Stochastic Weight Averaging (SWA)
- Aggregates weights along the SGD trajectory
- Improves robustness without explicitly training large ensembles
- Applied separately to control and stimulated models

### 3. Bayesian VEGA
- Replaces deterministic linear layers with Bayesian linear layers
- Learns posterior distributions over weights
- Provides principled uncertainty estimates via KL divergence regularization

---

## Experiments

- Models trained using:
  - Activator and inhibitor gene sets
  - Merged TF–gene regulatory masks
- Comparisons performed between:
  - Control vs stimulated regulatory programs
  - Deterministic, SWA, and Bayesian formulations
- Analysis focuses on identifying transcription factors and target genes exhibiting condition-specific regulatory rewiring and uncertainty
