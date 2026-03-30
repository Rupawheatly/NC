# Neural Collapse Dynamics  
### Depth, Activation, Regularisation, and Feature Norm Thresholds

Official code and experiment notebooks for the **Neural Collapse Dynamics** research project.  
This repository contains experiments for MNIST, FashionMNIST, and CIFAR-10 using MLP and ResNet architectures, including ablations, sweeps, and intervention studies.

---

## 📦 Repository Contents

### 📝 Jupyter Notebooks
| Notebook | Purpose |
|----------|---------|
| `Neural Collapse Dynamics.ipynb` | Overview notebook with experiment walkthrough |
| `neural-collapse-mnist-baseline.ipynb` | Baseline MLP-5 on MNIST, two-phase protocol |
| `nc-mnist-sweep.ipynb` | H1 depth sweep + H2 activation sweep on MNIST |
| `nc-mnist-remaining.ipynb` | GELU and Tanh seed studies |
| `nc-mnist-wd-sweep.ipynb` | H3 weight decay sweep on MNIST |
| `NC Baseline Weight Decay Sweep.ipynb` | Additional weight decay experiments for baseline |
| `NC_Width_Sweep_A100result.ipynb` | H5 width sweep on A100 GPU |
| `NC_Final_Colab_A100 with results.ipynb` | H4 norm threshold consolidation |
| `NC_Ablations with results.ipynb` | CE-only ablation + extra seeds |
| `NC_CIFAR10_Baseline (1).ipynb` | ResNet-20 baseline on CIFAR-10 |
| `NC_MLP5_CIFAR10 with result.ipynb` | MLP-5 architecture comparison on CIFAR-10 |
| `NC_ResNet20_MNIST result seed 0 only.ipynb` | ResNet-20 MNIST experiments for seed 0 |
| `NC_ResNet20_MNIST result seed 1 and 2.ipynb` | ResNet-20 MNIST experiments for seeds 1 and 2 |
| `NC_Intervention_Experiment_(1).ipynb` | Intervention experiment study |

---

### 📂 Folders
| Folder | Purpose |
|--------|---------|
| `Results/` | CSVs and data outputs for all experiments |
| `figures/` | Saved plots and visualizations from notebooks |

---

## 🚀 Installation & Requirements

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scipy seaborn


Tested with:
- Python 3.10
- PyTorch 2.1 (CUDA 11.8)
- torchvision 0.16

A **GPU is required** for full reproduction. All experiments were run on:
- Kaggle T4 (16 GB VRAM) — MNIST sweeps
- Google Colab A100 (40 GB VRAM) — width sweep, final consolidation
- Google Colab (T4) — CIFAR-10 experiments

CPU-only is possible for the baseline but will be slow.
### Running the baseline

Open `notebooks/neural-collapse-mnist-baseline.ipynb` on Kaggle or Colab with a GPU runtime. It will:

1. Download MNIST automatically via `torchvision.datasets`
2. Train MLP-5 (depth=5, width=512, ReLU) for 200 epochs with CE loss (Phase 1)
3. Switch to MSE loss for 300 epochs (Phase 2)
4. Compute NC1, NC2, NC3, and fn every 10 epochs
5. Save `mnist_twophase.csv` with the full training curve

Expected output:
```
T_NC (NC1<0.01): epoch 310  feat_norm=1.063
Test accuracy: 97.1%
```

### Running the full sweep

Each notebook is self-contained. Run them in order for full reproduction.
Each notebook saves its CSV outputs to the working directory (`/kaggle/working/` or `/tmp/`) and generates figures inline. The final summary figures in the paper were produced by `NC_Final_Colab_A100 with results.ipynb`, which loads the per-experiment CSVs and produces all panels.



## Acknowledgements

Experiments run on Kaggle (free T4 GPU) and Google Colab (free A100).  
Datasets downloaded via `torchvision.datasets` (MNIST, CIFAR-10).  
ResNet-20 implementation follows He et al. (2016), CIFAR-10 variant.
