# Supplementary Material

**Paper:** Neural Collapse Dynamics: Depth, Activation, Regularisation, and Feature Norm Thresholds
**Submitted to:** Neurocomputing (Elsevier)

Everything needed to reproduce every reported number and figure from the per-run
training logs, no GPU required. All notebooks include their executed outputs.

## Contents

- `notebooks/reproduce_all.ipynb` — **one notebook that reproduces the paper** (CPU): recomputes every reported number and regenerates every figure directly from the released logs. Run All.
- `notebooks/training/` — GPU notebooks that produced the data, including:
  - `E6_Optimizer_Deconfound.ipynb` — matched-optimiser de-confound (Table 8; +582% vs +458%).
  - `E7_ViT_CrossArch.ipynb` — ViT cross-architecture check (relaxed collapse; LayerNorm caveat).
  - `E8_EarlyWarning_Benchmark.ipynb` — multi-signal benchmark + multi-seed robustness (Appendix A).
  - plus the baseline, depth, activation, weight-decay, width, Fashion-MNIST,
    ResNet-20, intervention, and (failed) CIFAR-100 runs.
- `results/` — per-run, per-epoch CSVs; `results/vit_crossarch/`,
  `results/early_warning/`, and `results/deconfound/` hold the ViT, benchmark,
  and matched-optimiser de-confound runs.
- `figures/` — paper figures and graphical abstract (PNG).

## Data format (`results/*.csv`)

One row per logged epoch: `epoch`, `phase` (1 = CE, 2 = MSE), `train`/`test`
accuracy, `nc1`/`nc2`/`nc3`, `feat_norm` (fn). The `early_warning/` CSVs also log
`train_loss`, `train_err`, `weight_norm`, `grad_norm`; `vit_crossarch/` adds
`feat_norm_preln` (pre-LayerNorm norm). `*_summary.csv` files give per-seed fn* / T_NC. `results/deconfound/` holds the E6 per-seed fn* and summary transcribed from the E6 notebook's executed output.

## Reproduce

Open `notebooks/reproduce_all.ipynb` next to `../results/` and **Run All** (Colab:
set the Drive path in the first code cell). It recomputes and cross-checks every
number — sweeps, effect sizes, Kruskal–Wallis stats, the intervention, the temporal
lead (both the 8-run width-sweep lead and the 12-run strict-threshold lead,
75 +/- 51 epochs), and the ViT (E7) / 5-seed robustness / early-warning benchmark
(E8) additions — and regenerates all figures (including the graphical abstract and
de-confound figure as PNG + PDF). The **E6** de-confound numbers are reproduced in
`results/deconfound/` and in the E6 notebook's executed output.

## Conventions

Sample std (ddof = 1); 95% Student-t CIs (df = N − 1); rank-based Kruskal–Wallis
with ε² effect size. Training on single T4/A100 GPUs; verification on CPU.
