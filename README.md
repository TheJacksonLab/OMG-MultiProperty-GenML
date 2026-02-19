# OMG MultiProperty GenML

<p align="center">
<img src="https://github.com/TheJacksonLab/OMG-MultiProperty-GenML/blob/main/data/figure/TOC.png" width="1250" height="500">
</p>

### This repository contains python scripts and data for a generative model for multi-property polymer design.

# Setup
## Set up Python environment with Anaconda (A or B)
### A. Original conda environment used (Linux / NVIDIA)
```
conda env create -f environment.yml
```
### B. Conda environment for CPU
```
conda env create -f environment-CPU.yml
```

# Components
To run a script, a file path in the script should be modified to be consistent with an attempted directory.

### 1. ./train/data
This directory contains data to train a generative model.

### 2. ./train/fitting_to_polymer
This directory contains data for monomer-polymer property correlations.

### 3. ./train/group_selfies_vae_train
There is a trained model available under ./polymerization_step_growth_750k/

### 4. ./vae

### 5. ./group-selfies

## Authors
Seonghwan Kim, Charles M. Schroeder, and Nicholas E. Jackson



