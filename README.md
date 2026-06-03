# Supplementary Material

**Paper:** Neural Collapse Dynamics: Depth, Activation, Regularisation, and Feature Norm Thresholds
**Manuscript ID:** Access-2026-14281

This package lets a reader reproduce every reported quantity and figure in the
paper directly from the per-run training logs, without needing a GPU or re-running
training.

## Contents

```
Supplementary Material/
├── README.md
├── notebooks/
│   ├── verify_paper_numbers.ipynb          # recomputes every number in the paper from the CSVs
│   ├── generate_all_revision_figures.ipynb # regenerates all figures from the CSVs
│   ├── verify_csvs.ipynb                   # CSV integrity / fn* summary checker
│   └── training/                           # the notebooks that produced the data (require GPU)
│       ├── neural-collapse-mnist-baseline.ipynb
│       ├── nc-mnist-sweep.ipynb            # depth sweep (MNIST, MLP)
│       ├── nc-mnist-wd-sweep.ipynb         # weight-decay sweep
│       ├── NC Baseline Weight Decay Sweep.ipynb
│       ├── NC_Width_Sweep_A100result.ipynb # width sweep
│       ├── NC_CIFAR10_Baseline.ipynb       # ResNet-20 / CIFAR-10
│       ├── NC_FashionMNIST_Baseline.ipynb
│       ├── NC_ResNet20_MNIST result seed 0 only.ipynb
│       ├── NC_ResNet20_MNIST result seed 1 and 2.ipynb
│       ├── NC_Intervention_Experiment.ipynb        # feature-norm rescaling intervention
│       ├── NC_Final_Colab_A100 with results.ipynb
│       ├── E1_FashionMNIST_MLP5.ipynb              # Fashion-MNIST / MLP-5
│       ├── E4_Activation_Homogeneity.ipynb         # ReLU/LeakyReLU/Tanh/GELU
│       ├── E4b_SiLU_Extended.ipynb                 # SiLU plateau
│       ├── E5_FashionMNIST_ResNet20.ipynb          # Fashion-MNIST / ResNet-20
│       └── E2_CIFAR100_ResNet20*.ipynb             # CIFAR-100 attempts (documented as failures)
├── results/                                # per-run, per-epoch training logs (100 CSV files)
└── figures/                                # the 11 figures used in the paper (PNG)
```

All notebooks are shared **with their executed outputs**, so results are visible
without re-running. The three verification notebooks at the top of `notebooks/`
reproduce every reported number and figure from the saved CSVs and run on CPU;
the `training/` notebooks are the original training scripts that produced the
CSVs and require a GPU to re-run.

## Data format (`results/*.csv`)

Each CSV is one training run, with one row per logged epoch and the columns:

| column | meaning |
|---|---|
| `epoch` | training epoch |
| `phase` | 1 = CE phase, 2 = MSE phase (two-phase protocol) |
| `train`, `test` | accuracy |
| `nc1`, `nc2`, `nc3` | neural-collapse metrics |
| `feat_norm` | mean penultimate feature norm (fn) |

Summary files (`*_summary.csv`) hold per-seed `fn`/`T_NC` for runs logged at
summary granularity.

## How to reproduce

1. Open `notebooks/verify_paper_numbers.ipynb` (Google Colab or local Jupyter).
2. Point it at the `results/` folder (the notebook auto-detects a local `results/`
   directory, or set the Drive path in the first cell for Colab).
3. Run all cells. The notebook recomputes — and prints — every depth, activation,
   weight-decay, width, and (architecture × dataset) value, the Kruskal–Wallis
   statistics and effect sizes, the intervention results, the predictive lead time,
   and the CIFAR-100 diagnostics, with a built-in cross-check against the
   paper-reported values.
4. `generate_all_revision_figures.ipynb` regenerates every figure from the same CSVs.

## Conventions

- Dispersion statistics use the sample standard deviation (ddof = 1).
- Confidence intervals are 95% Student-t intervals (df = N − 1).
- Between-pair significance uses the rank-based Kruskal–Wallis test with the
  ε² effect size.

## Hardware

All training runs used single NVIDIA T4 GPUs (Google Colab and Kaggle); the
verification notebooks need only CPU. The same code is also mirrored at:
https://github.com/Rupawheatly/NC
