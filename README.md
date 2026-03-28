# Neural Collapse Dynamics

**Neural Collapse Dynamics: Depth, Activation, Regularisation, and Norm Thresholds in Neural Collapse**

---

## Overview

This repository contains all experiments for our study of *when* and *how fast* neural collapse (NC) emerges as a function of architecture and training hyperparameters — a question that has received little systematic attention despite NC being well understood at equilibrium.

Our central finding: **NC timing is predicted by a (model, dataset)-specific feature norm threshold.** Training conditions control the time to collapse; the feature norm at collapse concentrates tightly within each (model, dataset) pair (CV < 6%) and is predictive of collapse onset across widely varying conditions.

---

## Key Results

| Experiment | Finding |
|---|---|
| Depth sweep (d = 2, 3, 5, 7) | Non-monotonic effect — depth-3 fastest (T_NC = 250), depth-7 slowest (340) |
| Activation (ReLU / GELU / Tanh) | Each activation defines a different fn\* — ReLU 1.077, Tanh 1.325, GELU 2.128 |
| Weight decay (λ = 1e-5 → 5e-4) | λ controls approach speed; fn\* stable at CV = 5.6% over 10× range |
| Feature norm threshold | MNIST MLP-5: 1.052 ± 0.063 · CIFAR ResNet-20: 1.515 ± 0.007 |
| Architecture vs dataset | Architecture effect (+68%) is 4× larger than dataset effect (−14%) |
| Width sweep (128 → 1024) | fn\* shifts by only 13% over 8× width range; width mainly controls speed |

---

## Notebooks

| Notebook | GPU | What it runs |
|---|---|---|
| `neural-collapse-mnist-baseline` | T4 | MLP-5 MNIST two-phase baseline (seed 0) |
| `nc-mnist-sweep` | T4 | Depth sweep (d=2,3,7) + GELU seeds 0–1 |
| `nc-mnist-remaining` | T4 | Depth-2,3 seeds · Tanh · GELU seeds 1–2 |
| `nc-mnist-wd-sweep` | T4 | Weight decay sweep λ ∈ {1e-5, 5e-5, 1e-4, 5e-4} |
| `NC_Final_Colab_A100` | A100 80GB | Depth-5 seeds 1–2 · threshold table (N=12+3) |
| `NC_Ablations` | A100 40GB | CE-only ablation · GELU seeds 3–4 |
| `NC_CIFAR10_Baseline` | A100 80GB | ResNet-20 CIFAR-10 (3 seeds) |
| `NC_MLP5_CIFAR10` | A100 40GB | MLP-5 CIFAR-10 — architecture vs dataset test |
| `NC_Width_Sweep_A100` | A100 40GB | Width sweep 128 / 256 / 512 / 1024 × 3 seeds |

---

## Setup
```bash
pip install torch torchvision numpy pandas matplotlib scipy
```

All notebooks are self-contained. Each downloads MNIST or CIFAR-10 automatically and saves results to CSV.

**Two-phase protocol:** Phase 1 trains with cross-entropy for 200 epochs to reach ≥99% train accuracy. Phase 2 switches to MSE loss to drive NC1 collapse. CE alone does not reach NC1 < 0.01 within 600 epochs in our setup.

---


