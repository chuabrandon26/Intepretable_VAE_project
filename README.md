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

---
## The Script

This is the core project logic used in this repository. It covers: loading PBMC data, building an interpretable TF→gene decoder mask from CollecTRI, training VEGA models separately for **stimulated** vs **control**, and exporting TF rewiring weights (mean/std or Bayesian uncertainty) for downstream plotting.

```python
# ----------------------------
# 0) Imports + device
# ----------------------------
import os
import math
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils
import decoupler as dc
import scanpy as sc

device = "cuda"

# ----------------------------
# 1) Load PBMC train/valid (h5ad)
# ----------------------------
# download PBMC dataset for training
!wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1zHJKoU8QcQB4cLR-oICO2YY4Nu-QaZHG" -O PBMCtrain.h5ad
PBMCtrain = sc.read_h5ad("PBMCtrain.h5ad")

# for validation
!wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1rJKZYIG7rv7BQbDD9RElYOgHcxxzjcYj" -O PBMCvalid.h5ad
PBMCvalid = sc.read_h5ad("PBMCvalid.h5ad")

# ----------------------------
# 2) TF prior network (CollecTRI) + filter TFs (IFN-related + extras)
# ----------------------------
regulons = dc.get_collectri(organism="human", split_complexes=False)

filteredregulons = regulons[regulons["source"].isin([
    "GTF2I","GTF3A","NRF1","ELF1",
    "STAT1","STAT2","IRF9","STAT3","STAT4","STAT5A","STAT5B",
    "IRF3","IRF7","IRF1","IRF5","IRF8"
])]

# ----------------------------
# 3) Split into stimulated vs control (train + validation)
# ----------------------------
PBMCtrainstim  = PBMCtrain[PBMCtrain.obs["condition"] == "stimulated"]
PBMCtrainctrl  = PBMCtrain[PBMCtrain.obs["condition"] == "control"]
PBMCvalidstim  = PBMCvalid[PBMCvalid.obs["condition"] == "stimulated"]
PBMCvalidctrl  = PBMCvalid[PBMCvalid.obs["condition"] == "control"]

print("PBMCtrainstim", len(PBMCtrainstim))
print("PBMCtrainctrl", len(PBMCtrainctrl))
print("PBMCvalidstim", len(PBMCvalidstim))
print("PBMCvalidctrl", len(PBMCvalidctrl))

# ----------------------------
# 4) Build VEGA mask (genes x TFs) from filtered regulons
# ----------------------------
def createmask(pbmctrain, pbmcval, filteredregulons, extranodescount=1):
    filteredregulons = filteredregulons.astype({"source": str, "target": str, "weight": np.float64})

    tmp = (
        pl.from_pandas(filteredregulons)
        .with_columns(pl.col("weight").replace(-1, 1))     # convert -1 to 1
        .filter(pl.col("target").is_in(pbmctrain.var.index.to_numpy()))
        .pivot(on="source", index="target", values="weight")
        .fill_null(0)
    )

    # add extra unannotated node(s)
    for i in range(extranodescount):
        nodename = f"unannotated_{i+1}"
        tmp = tmp.with_columns(pl.lit(1).alias(nodename))

    train = pbmctrain[:, tmp["target"].to_list()].copy()
    valid = pbmcval[:, tmp["target"].to_list()].copy()
    return tmp, train, valid

# ----------------------------
# 5) VEGA model: encoder + masked sparse decoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, latentdims, inputdims, dropout, zdropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputdims, 800),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.mu = nn.Sequential(nn.Linear(800, latentdims), nn.Dropout(p=zdropout))
        self.sigma = nn.Sequential(nn.Linear(800, latentdims), nn.Dropout(p=zdropout))

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))  # numeric stability
        z = mu + sigma * self.N.sample(mu.shape)  # reparameterization trick
        self.kl = (0.5 * (sigma**2) + 0.5 * (mu**2) - torch.log(sigma) - 0.5).sum()
        return z

class SparseLayer(nn.Module):
    def __init__(self, mask, softpenalty):
        super().__init__()
        self.mask = nn.Parameter(torch.tensor(mask, dtype=torch.float).t(), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(mask.shape, mask.shape))[1]
        self.bias = nn.Parameter(torch.Tensor(mask.shape))[1]
        self.softpenalty = softpenalty
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.weight.data = self.weight.data * self.mask

    def forward(self, input):
        w = self.weight * self.mask
        out = input.mm(w.t())
        out = out + self.bias.unsqueeze(0).expand_as(out)
        return out

class DecoderVEGA(nn.Module):
    def __init__(self, mask, softpenalty):
        super().__init__()
        self.sparselayer = nn.Sequential(SparseLayer(mask, softpenalty))

    def forward(self, x):
        return self.sparselayer(x.to(device))

    def positiveweights(self):
        w = self.sparselayer.weight
        w.data = w.data.clamp(0)
        return w

class VEGA(nn.Module):
    def __init__(self, latentdims, inputdims, mask, dropout=0.3, zdropout=0.3):
        super().__init__()
        self.encoder = Encoder(latentdims, inputdims, dropout, zdropout)
        self.decoder = DecoderVEGA(mask, softpenalty=0.1)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ----------------------------
# 6) Training loops (vanilla VEGA, SWA, Bayesian decoder)
# ----------------------------
def trainVEGAwithvalid(vae, data, valdata, epochs=100, beta=0.0001, learningrate=0.001):
    opt = torch.optim.Adam(vae.parameters(), lr=learningrate, weight_decay=5e-4)
    trainlosses, validlosses = [], []

    for epoch in range(epochs):
        trainlosse = 0
        vae.train()

        for x in data:
            x = x.to(device)
            opt.zero_grad()
            xhat = vae(x)
            loss = (x - xhat).pow(2).sum() + beta * vae.encoder.kl
            loss.backward()
            opt.step()
            vae.decoder.positiveweights()
            trainlosse += loss.detach().cpu().numpy()

        trainlosses.append(trainlosse / (len(data) * 128))

        vae.eval()
        validlosse = 0
        for x in valdata:
            x = x.to(device)
            xhat = vae(x)
            loss = (x - xhat).pow(2).sum() + beta * vae.encoder.kl
            validlosse += loss.detach().cpu().numpy()

        validlosses.append(validlosse / (len(valdata) * 128))

        if epoch % 10 == 0:
            print("epoch", epoch,
                  "trainloss", trainlosses[-1],
                  "validloss", validlosses[-1])

    return vae, trainlosses, validlosses

def getweight(vae):
    vae.eval()
    return vae.decoder.sparselayer.weight.data.cpu().numpy()

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

def trainVEGAwithswa(vae, data, valdata, epochs=100, beta=0.0001,
                     learningrate=0.001, swastart=75, swalr=0.05):
    opt = torch.optim.Adam(vae.parameters(), lr=learningrate, weight_decay=5e-4)
    swamodel = AveragedModel(vae)
    swascheduler = SWALR(opt, swa_lr=swalr)

    trainlosses, validlosses = [], []

    for epoch in range(epochs):
        vae.train()
        trainlosse = 0.0

        for x in data:
            x = x.to(device)
            opt.zero_grad()
            xhat = vae(x)
            loss = (x - xhat).pow(2).sum() + beta * vae.encoder.kl
            loss.backward()
            opt.step()
            vae.decoder.positiveweights()
            trainlosse += loss.detach().cpu().item()

        if epoch >= swastart:
            swamodel.update_parameters(vae)
            swascheduler.step()

        trainlosses.append(trainlosse / len(data))

        vae.eval()
        validlosse = 0.0
        with torch.no_grad():
            for x in valdata:
                x = x.to(device)
                xhat = vae(x)
                loss = (x - xhat).pow(2).sum() + beta * vae.encoder.kl
                validlosse += loss.detach().cpu().item()

        validlosses.append(validlosse / len(valdata))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: trainloss={trainlosses[-1]:.4f}, validloss={validlosses[-1]:.4f}")

    update_bn(data, swamodel, device=device)
    return swamodel, trainlosses, validlosses

def getweightswa(swamodel):
    model = swamodel.module if hasattr(swamodel, "module") else swamodel
    return model.decoder.sparselayer.weight.data.cpu().numpy()

# Bayesian decoder version (stores mu and log sigma and exports uncertainty)
def klloss(mu0, logsigma0, mu1, logsigma1):
    kl = logsigma1 - logsigma0 + (torch.exp(logsigma0)**2 + (mu0 - mu1)**2) / (2 * math.exp(logsigma1)**2) - 0.5
    return kl.sum()

class BayesianSparseLayer(nn.Module):
    def __init__(self, mask, softpenalty):
        super().__init__()
        self.mask = nn.Parameter(torch.tensor(mask, dtype=torch.float).t(), requires_grad=False)
        self.infeatures = mask.shape
        self.outfeatures = mask.shape[1]

        self.priormu = 0
        self.priorsigma = 0.1
        self.priorlogsigma = math.log(0.1)

        self.weightmu = nn.Parameter(torch.Tensor(self.outfeatures, self.infeatures))
        self.weightlogsigma = nn.Parameter(torch.Tensor(self.outfeatures, self.infeatures))
        self.biasmu = nn.Parameter(torch.Tensor(self.outfeatures))
        self.biaslogsigma = nn.Parameter(torch.Tensor(self.outfeatures))

        self.softpenalty = softpenalty
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weightmu.size(1))
        self.weightmu.data.uniform_(-stdv, stdv)
        self.weightlogsigma.data.fill_(self.priorlogsigma)
        self.biasmu.data.uniform_(-stdv, stdv)
        self.biaslogsigma.data.fill_(self.priorlogsigma)

        self.weightmu.data = self.weightmu.data * self.mask
        self.weightlogsigma.data = self.weightlogsigma.data * self.mask

    def forward(self, input):
        weight = self.weightmu + torch.exp(self.weightlogsigma) * torch.randn_like(self.weightlogsigma)
        bias = self.biasmu + torch.exp(self.biaslogsigma) * torch.randn_like(self.biaslogsigma)
        w = weight * self.mask
        out = input.mm(w.t())
        out = out + bias.unsqueeze(0).expand_as(out)
        return out

    def klloss(self):
        return klloss(self.weightmu, self.weightlogsigma, self.priormu, self.priorlogsigma)

class BayesDecoder(nn.Module):
    def __init__(self, mask, softpenalty):
        super().__init__()
        self.sparselayer = nn.Sequential(BayesianSparseLayer(mask, softpenalty))

    def forward(self, x):
        return self.sparselayer(x.to(device))

    def kldivergence(self):
        return self.sparselayer.klloss()

    def positiveweights(self):
        w = self.sparselayer.weightmu
        w.data = w.data.clamp(0)
        return w

class BayesVEGA(nn.Module):
    def __init__(self, latentdims, inputdims, mask, dropout=0.3, zdropout=0.3, softpenalty=0.1):
        super().__init__()
        self.encoder = Encoder(latentdims, inputdims, dropout, zdropout)
        self.decoder = BayesDecoder(mask, softpenalty)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def getweightbayes(model):
    return model.decoder.sparselayer.weightmu.data.cpu().numpy()

def getweightuncertaintiesbayes(model):
    return model.decoder.sparselayer.weightlogsigma.data.cpu().numpy()

def trainVEGAwithbayes(vae, data, valdata, epochs=100, betaen=0.0001, betade=0.0001, learningrate=0.001):
    opt = torch.optim.Adam(vae.parameters(), lr=learningrate, weight_decay=5e-4)
    trainlosses, validlosses = [], []

    for epoch in range(epochs):
        trainlosse = 0
        vae.train()

        for x in data:
            x = x.to(device)
            opt.zero_grad()
            xhat = vae(x)

            klencoder = vae.encoder.kl
            kldecoder = vae.decoder.kldivergence()

            loss = (x - xhat).pow(2).sum() + betaen * klencoder + betade * kldecoder
            loss.backward()
            opt.step()

            vae.decoder.positiveweights()
            trainlosse += loss.detach().cpu().numpy()

        trainlosses.append(trainlosse / (len(data) * 128))

        vae.eval()
        validlosse = 0
        for x in valdata:
            x = x.to(device)
            xhat = vae(x)
            klencoder = vae.encoder.kl
            kldecoder = vae.decoder.kldivergence()
            loss = (x - xhat).pow(2).sum() + betaen * klencoder + betade * kldecoder
            validlosse += loss.detach().cpu().numpy()

        validlosses.append(validlosse / (len(valdata) * 128))

        if epoch % 10 == 0:
            print("epoch", epoch,
                  "trainloss", trainlosses[-1],
                  "validloss", validlosses[-1])

    return vae, trainlosses, validlosses

# ----------------------------
# 7) Run wrapper: train N runs and export mean/std (or Bayesian uncertainty)
# ----------------------------
def savelosses(trainlosses, validlosses, path, condition):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{condition}_losses.csv"), "w") as f:
        f.write("epoch,trainloss,validloss\n")
        for epoch, (trainloss, validloss) in enumerate(zip(trainlosses, validlosses)):
            f.write(f"{epoch},{trainloss},{validloss}\n")

def runvegamodel(modeltype, traindata, validdata, pathtosave="modelsvega",
                cond="all", celltype="all", N=10, epochs=60):

    maskdf, traindata, validdata = createmask(traindata, validdata, filteredregulons)

    numericcolumns = [
        name for name, dtype in maskdf.schema.items()
        if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    mask = maskdf.select(numericcolumns).to_numpy()
    mask = torch.from_numpy(mask).float()

    if cond != "all":
        traindata = traindata[traindata.obs["condition"] == cond]
        validdata = validdata[validdata.obs["condition"] == cond]

    if celltype != "all":
        traindata = traindata[traindata.obs["celltype"] == celltype]
        validdata = validdata[validdata.obs["celltype"] == celltype]

    trainX = torch.utils.data.DataLoader(traindata.X.toarray(), batch_size=128, shuffle=True)
    validX = torch.utils.data.DataLoader(validdata.X.toarray(), batch_size=128, shuffle=True)

    allweights = []

    if modeltype == "vega":
        for _ in range(N):
            os.makedirs(pathtosave, exist_ok=True)
            vega = VEGA(latentdims=mask.shape, inputdims=mask.shape, mask=mask.T,[1]
                        dropout=0.3, zdropout=0.3).to(device)
            vega, trainlosses, validlosses = trainVEGAwithvalid(vega, trainX, validX, epochs=epochs, beta=0.0001)
            allweights.append(getweight(vega))
            savelosses(trainlosses, validlosses, pathtosave, cond)

    elif modeltype == "swa":
        for _ in range(N):
            os.makedirs(pathtosave, exist_ok=True)
            vega = VEGA(latentdims=mask.shape, inputdims=mask.shape, mask=mask.T,[1]
                        dropout=0.3, zdropout=0.3).to(device)
            swamodel, trainlosses, validlosses = trainVEGAwithswa(
                vega, trainX, validX,
                epochs=100, beta=0.0001, learningrate=0.0005, swastart=80, swalr=0.0001
            )
            allweights.append(getweightswa(swamodel))
            savelosses(trainlosses, validlosses, pathtosave, cond)

    elif modeltype == "bayes":
        weightuncertainties = []
        for _ in range(N):
            os.makedirs(pathtosave, exist_ok=True)
            vega = BayesVEGA(latentdims=mask.shape, inputdims=mask.shape, mask=mask.T,[1]
                             dropout=0.3, zdropout=0.3).to(device)
            vega, trainlosses, validlosses = trainVEGAwithbayes(
                vega, trainX, validX,
                epochs=epochs, betaen=0.0001, betade=0.0001
            )
            allweights.append(getweightbayes(vega))
            unc = np.exp(getweightuncertaintiesbayes(vega))
            weightuncertainties.append(unc)
            savelosses(trainlosses, validlosses, pathtosave, cond)

        weightuncertainties = np.stack(weightuncertainties)
        meanunc = weightuncertainties.mean(axis=0)
        meanunc = pd.DataFrame(meanunc, index=traindata.var.index.tolist(), columns=maskdf.columns[1:])
        meanunc.to_csv(os.path.join(pathtosave, f"{cond}_{modeltype}_{celltype}_weightuncertaintymean.csv"), index=True)

    stacked = np.stack(allweights)
    meanweight = stacked.mean(axis=0)
    stdweight = stacked.std(axis=0)

    meanweight = pd.DataFrame(meanweight, index=traindata.var.index.tolist(), columns=maskdf.columns[1:])
    stdweight  = pd.DataFrame(stdweight,  index=traindata.var.index.tolist(), columns=maskdf.columns[1:])

    meanweight.to_csv(os.path.join(pathtosave, f"{cond}_{modeltype}_{celltype}_weightmean.csv"), index=True)
    stdweight.to_csv(os.path.join(pathtosave, f"{cond}_{modeltype}_{celltype}_weightstd.csv"), index=True)

    return meanweight, stdweight

# Example: train stimulated vs control (vanilla VEGA)
mean_stim, std_stim = runvegamodel("vega", PBMCtrainstim, PBMCvalidstim, pathtosave="modelsvega",
                                  cond="stimulated", celltype="all", N=1, epochs=60)
mean_ctrl, std_ctrl = runvegamodel("vega", PBMCtrainctrl, PBMCvalidctrl, pathtosave="modelsvega",
                                  cond="control", celltype="all", N=1, epochs=60)

# ----------------------------
# 8) Plot TF rewiring: control vs stimulated (with error bars)
# ----------------------------
def plottfrewiring(path, tfnames, title, ncols=3, figsize=(15, 12)):
    for root, _, files in os.walk(path):
        for filename in files:
            print("Processing file:", filename)

            if "bayes" in filename:
                if filename.endswith("uncertaintymean.csv"):
                    if "control" in filename:
                        sigmaweightsctrl = pd.read_csv(os.path.join(root, filename), index_col=0)
                    elif "stimulated" in filename:
                        sigmaweightsstim = pd.read_csv(os.path.join(root, filename), index_col=0)
            else:
                if filename.endswith("std.csv"):
                    if "control" in filename:
                        sigmaweightsctrl = pd.read_csv(os.path.join(root, filename), index_col=0)
                    elif "stimulated" in filename:
                        sigmaweightsstim = pd.read_csv(os.path.join(root, filename), index_col=0)

            if filename.endswith("weightmean.csv"):
                if "control" in filename:
                    meanweightsctrl = pd.read_csv(os.path.join(root, filename), index_col=0)
                elif "stimulated" in filename:
                    meanweightsstim = pd.read_csv(os.path.join(root, filename), index_col=0)

    nplots = len(tfnames)
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=16)

    for idx, tfname in enumerate(tfnames):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]

        stim = meanweightsstim[tfname].rename("stim")
        ctrl = meanweightsctrl[tfname].rename("ctrl")

        stdstim = sigmaweightsstim[tfname].rename("stdstim")
        stdctrl = sigmaweightsctrl[tfname].rename("stdctrl")

        merged = pd.concat([ctrl, stim, stdctrl, stdstim], axis=1).dropna()

        ax.errorbar(
            merged["ctrl"], merged["stim"],
            xerr=merged["stdctrl"], yerr=merged["stdstim"],
            fmt="o", ecolor="black", elinewidth=0.5, markersize=5,
            markeredgecolor="blue", markeredgewidth=0.5
        )

        ax.plot(
            [merged["ctrl"].min(), merged["ctrl"].max()],
            [merged["ctrl"].min(), merged["ctrl"].max()],
            color="grey", linestyle="--"
        )

        ax.set_xlabel("Control mean weight")
        ax.set_ylabel("Stimulated mean weight")
        ax.set_title(f"{tfname} weight control vs. stimulated")

        merged["dist"] = np.abs(merged["stim"] - merged["ctrl"]) / np.sqrt(2)
        top = merged.nlargest(20, "dist")
        for gene, r in top.iterrows():
            ax.annotate(gene, (r["ctrl"], r["stim"]), fontsize=8)

    plt.tight_layout()
    plt.show()

# Example plot
plottfrewiring(
    path="modelsvega",
    tfnames=["STAT1", "STAT2"],
    title="TF Rewiring Control vs Stimulated"
)

