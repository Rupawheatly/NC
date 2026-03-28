Neural Collapse Dynamics: Depth, Activation, Regularisation, and Feature Norm Thresholds
Official code and data for the paper:

Neural Collapse Dynamics: Depth, Activation, Regularisation, and Feature Norm Thresholds
[Author Names] · IEEE Access · 2026
[Paper] · [arXiv]


What this repository contains
├── notebooks/                        # All experiment notebooks (run on Kaggle/Colab)
│   ├── neural-collapse-mnist-baseline.ipynb      # Baseline: MLP-5, MNIST, two-phase protocol
│   ├── nc-mnist-sweep.ipynb                      # H1 depth sweep + H2 activation sweep
│   ├── nc-mnist-remaining.ipynb                  # GELU seeds 1–2, Tanh seeds 0–2
│   ├── nc-mnist-wd-sweep.ipynb                   # H3 weight decay sweep
│   ├── NC_Width_Sweep_A100result.ipynb           # H5 width sweep (A100)
│   ├── NC_Final_Colab_A100 with results.ipynb    # H4 feature norm threshold consolidation
│   ├── NC_CIFAR10_Baseline (1).ipynb             # ResNet-20 / CIFAR-10 baseline
│   ├── NC_MLP5_CIFAR10 with result.ipynb         # MLP-5 / CIFAR-10 (architecture effect)
│   ├── NC_Ablations with results.ipynb           # CE-only ablation + GELU seeds 3–4
│   └── NC Baseline Weight Decay Sweep.ipynb      # Early CIFAR-10 weight decay sweep
│
├── NC_IEEE_Access.tex                # Final IEEE Access LaTeX source
└── nc_references.bib                 # BibTeX bibliography

Key concepts
Neural collapse (NC) is a geometric regularity that emerges at the end of training: penultimate-layer features converge to a simplex equiangular tight frame (ETF), within-class variability vanishes (NC1), and classifier weights align with class means (NC3).
This paper asks: what controls when NC emerges?
Central finding: the mean feature norm (fn) at the moment of collapse concentrates tightly within each (model, dataset) pair — CV < 6% across depths, weight decays, widths, and seeds — and is predictive of collapse timing. Architecture family is the primary driver of this value (4× larger effect than dataset).
NC metrics measured:
MetricFormulaMeaningNC1Tr(Σ_W) / Tr(Σ_B) → 0Within-class collapseNC2mean|cos(mᵢ, mⱼ) + 1/(K−1)| → 0ETF alignmentNC31 − (1/K)Σ cos(ŵ_c, m̂_c) → 0Weight–feature dualityfn(1/N) Σ ‖hᵢ‖₂Mean feature norm
T_NC = first epoch where NC1 < ε (0.01 for ReLU networks; 0.05 for GELU, Tanh, and shallow depths ≤ 3).

Reproducing the results
Requirements
bashpip install torch torchvision numpy pandas matplotlib scipy
Tested with:

Python 3.10
PyTorch 2.1 (CUDA 11.8)
torchvision 0.16

A GPU is required for full reproduction. All experiments were run on:

Kaggle T4 (16 GB VRAM) — MNIST sweeps
Google Colab A100 (40 GB VRAM) — width sweep, final consolidation
Google Colab (T4) — CIFAR-10 experiments

CPU-only is possible for the baseline but will be slow (~4–6 hours per run vs ~20 minutes on T4).
Running the baseline
Open notebooks/neural-collapse-mnist-baseline.ipynb on Kaggle or Colab with a GPU runtime. It will:

Download MNIST automatically via torchvision.datasets
Train MLP-5 (depth=5, width=512, ReLU) for 200 epochs with CE loss (Phase 1)
Switch to MSE loss for 300 epochs (Phase 2)
Compute NC1, NC2, NC3, and fn every 10 epochs
Save mnist_twophase.csv with the full training curve

