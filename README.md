# Generative Multi-Property Refinement of Polymer Chemistries
A Generative Machine Learning Framework for Multi‑Property Polymer Design
<p align="center">
<img src="https://github.com/TheJacksonLab/OMG-MultiProperty-GenML/blob/main/TOC.png" width="1250" height="500">
</p>

## Overview
This repository contains Python scripts, trained models, and datasets for generative modeling workflow designed for multi-property refinement of polymer chemistries.
The framework integrates monomer-polymer property correlations, variational autoencoders (VAEs), and Bayesian optimization.

## 📦 Environment Setup
Set up a Python environment using Anaconda. Two options are provided depending on your preference:

### A. GPU Environment Used in This Work (Linux / NVIDIA)
```
conda env create -f environment.yml
```
### B. CPU-Only Environment
```
conda env create -f environment-CPU.yml
```

## 🗂 Repository Structure
Before running any script, update file paths inside the script to match your working directory.

### 1. ./train/data
Contains training datasets for the generative model.

### 2. ./train/fitting_to_polymer
Includes datasets and scripts constructing monomer–polymer property correlations.

### 3. ./train/group_selfies_vae_train
Holds a trained model and scripts for training and running the generative VAE.

### 4. ./vae
Core VAE components and supportive utility functions.

### 5. ./group-selfies
The original Group-SELFIES repository is required for model execution: https://github.com/aspuru-guzik-group/group-selfies. Clone it into this directory before running training scripts. 

## Authors
Seonghwan Kim, Charles M. Schroeder, and Nicholas E. Jackson
