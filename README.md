# Mini-batch Adaptive Partial Optimal Transport — experiments

Code for choosing the transported mass `s` of mini-batch partial OT (m-POT) per mini-batch,
and for testing that choice on domain adaptation, partial DA, generative modelling, gradient
flow and color transfer.  Protocols follow m-POT (Nguyen et al., ICML 2022) and its reference
code, so the m-POT numbers are directly comparable baselines.

## Layout

```
src/apot/        core: solvers, mass rules, class-overlap math, run bookkeeping
src/apps.py      argument parsers + run names of every trainer (shared with the launcher)
src/DeepDA/      digits.py (SVHN/USPS/MNIST) and office.py (Office-Home, VisDA, partial DA)
src/DeepGM/      train.py (CIFAR-10, CelebA; FID in memory)
src/GradientFlow/flow.py      src/ColorTransfer/transfer.py
src/analysis/    overlap_table.py, selector_sim.py, collect.py
scripts/         experiments.py (the plan), launch.py (GPU job queue)
tests/           test_apot.py (identities the rules rely on)
results/         all outputs (git-ignored)
```

Environment: conda env `mkpg-ot` (torch 2.11 + CUDA 12.8, POT 0.9.6).  Data under `data/`
as shipped (Office-Home / VisDA lists with `/data/...` paths resolved from the repo root).

## Mass rules (`--rule`)

| rule | mass of each mini-batch pair | needs |
|---|---|---|
| `ot` / `uot` | 1 (m-OT) / implicit (KL m-UOT, `--eps --tau`) | – |
| `fixed` / `ramp` | `--mass` (m-POT) / linear ramp to `--mass` | – |
| `elbow` | Kneedle knee of the POT cost curve (draft Alg. 1); `--elbow_mode chord` = same knee in 2 solves | – |
| `curv` | max 2nd difference of the cost curve (draft Alg. 2) | – |
| `cluster` | 1 − TV of K-means cluster histograms (draft Alg. 3) | features |
| `expected` | E[class overlap] for this batch size; `--prior uniform\|ema\|true` | (predictions) |
| `pseudo` | per-batch class overlap with predicted target labels (`--warmup`) | predictions |
| `price` | every pair cheaper than λ = κ·EMA(median C); `--target_mass` adapts λ by dual ascent | – |
| `oracle` | per-batch class overlap with TRUE target labels — diagnostic only | target labels |
| `aot` | exact AOT (Yang, Tan & Wen 2025) = OT on min(C, 0), our re-implementation | predictions |
| `ppot` | EMA of the confident-target fraction (PPOT, universal DA) | predictions |

Target labels are read only by `oracle` and by the logged diagnostics (batch overlap,
cross-class mass); no other rule sees them.

## Running

```bash
conda activate mkpg-ot
python tests/test_apot.py                          # 8 identity checks, ~5 s
python scripts/launch.py SMOKE                     # every trainer end to end, ~2 min
python src/analysis/overlap_table.py               # E[overlap] vs m-POT's tuned s (no training)
python src/analysis/selector_sim.py                # rules on synthetic batches (no training)

python scripts/launch.py E1 --status               # plan, state, GPU-hour estimate
nohup python scripts/launch.py E1 --gpus 0,1,2 > results/_launcher/E1.out 2>&1 &
python src/analysis/collect.py                     # tables -> results/analysis/tables.md
```

Filters: `--only oh:,visda:` / `--skip :curv:` (substrings of job labels), `--seeds 1`,
`--with aot,celeba`.  `--dry` prints the commands.

| exp | what | jobs | ≈ wall time on 3× RTX 5080 |
|---|---|---|---|
| E1 | go/no-go: per-batch mass vs best fixed mass (VisDA m=36/72; 4 PDA tasks incl. exact vs entropic m-POT and warm-up ablations; 2 seeds, seed-major) | 76 | 20 h |
| E2 | DA tables: Digits, VisDA, Office-Home (11 methods + published-solver m-POT, 3 seeds) | 540 | 247 h (Office-Home 1 seed: 62 h) |
| E3 | partial DA, Office-Home 65→25, 12 tasks (entropic ε = 0.5) | 396 | 118 h |
| E4 | fixed `s` at each finished adaptive run's mean mass | derived | – |
| E5 | CIFAR-10 generative model (+ CelebA with `--with celeba`) | 24 | 8 h |
| E6 | gradient flow + color transfer (CPU) | 45 | 2 h |
| PDACHECK | m-POT partial-DA reproduction on A2C, C2A, P2R (exact and entropic) | 6 | 1 h |

Suggested order: SMOKE → analysis → E1 → decide which methods carry the paper → E2/E3
subsets → E4 → E5/E6.

## Protocol notes (differences from the reference code are deliberate)

* Exact LP (`--eps 0`) for every partial/balanced rule; m-UOT uses m-POT's (ε, τ).  `mpot_ent` re-runs
  m-POT with its published solver (digits ε 0.1, VisDA ε 0.004, PDA ε 0.5; Office-Home's official
  setting is already exact) as a reproduction check.
* Settings were checked against the official GitHub code, not the local `~/Ada` copy, which carries
  local edits (digits 5 epochs instead of 100, PDA 500 iterations instead of 5000, ResNet V2 weights).
  Digits: 100 epochs, m 500, lr 4e-4, eta 0.1 / 0.1, 10 source-only epochs.
* ResNet-50 ImageNet **V1** weights (as the m-KOT runs); Office-Home 10-crop evaluation,
  VisDA single-crop every 1000 steps (10-crop on 55k images costs ~4 h per run).
* **Partial DA = the official m-POT `PartialDA` code** (`run_mOT.py`, `sh/exp_m*.sh`, checked
  against GitHub on 2026-10-11): target = the **alphabetically-first 25 class names** (identical,
  file by file, to the reference `*_25_list.txt`; the shipped lists number classes differently,
  Drill = 0, so "label < 25" would pick another subset); 5000 steps, single-crop eval every 500;
  `--aug crop` (Resize 256 + RandomCrop 224, not RandomResizedCrop); bottleneck 256; per-method
  cost weights: m-POT 0.03 / 7.5 / 1, m-UOT 0.003 / 0.75 / 10 (ε 0.01, τ 0.06), m-OT
  0.001 / 0.0001 / 1.  m-POT's mass follows the official ramp `mass/T·i + 0.01` then `mass/2`
  with `--mass 0.65`, i.e. 0.01 → 0.325 over the first 2500 steps (`--rule ramp --mass 0.325`);
  `mpot` solves it exactly, `mpot_ent` with the paper's entropic ε = 0.5.
* A fixed mass from step 0 collapses on PDA (source-private classes take the mass while the
  head is untrained), so PDA adaptive rules use `--warm_frac 0.5` (mass × min(1, t/2500), the
  ramp's shape); `oracle_nw` / `exp_ema_nw` / `mpot_fixed` are the no-warm-up ablations.
* PDA solver: every partial rule uses the published entropic ε = 0.5 (`PDA_EPS` in
  `scripts/experiments.py`; it beat the exact LP by 1–3 points on A2C / C2A / P2R); `mpot` keeps
  the exact LP as the solver ablation.  Costs are passed to POT in float64: the official script
  passes float32, where `exp(-M/0.5)` underflows at initialisation and POT returns NaN plans from
  step 0 (and it also uses Python 2's `iter.next()`), so the released code does not run as-is
  here.  PDACHECK (best accuracy, A2C / C2A / P2R): ours exact 60.0 / 69.2 / 79.7, ours entropic
  61.0 / 71.9 / 80.9, official code with only those two lines fixed 61.1 / 70.0 / 81.3, paper
  64.6 / 76.4 / 87.6.  Our port matches the official code; neither reaches the paper here, so
  the m-POT baseline is our re-run.  The official code also takes `log(softmax)` without an
  epsilon: once predictions are confident, float32 probabilities hit 0 and the cost turns NaN
  (C2A and P2R collapsed at ~2000 steps; best-accuracy reporting hides it).  Ours uses
  `log(p + 1e-10)`.
* Gradient flow uses the same (k, m) for every rule (the reference compared m-OT at m = 16
  exact against m-POT at m = 4 entropic).
* Color transfer uses the partial barycentric map (untransported colors keep their value) and
  reports W2 to the target palette, luminance SSIM and coverage.
* DGM: latent 32 and exact LP as in the m-KOT runs; the run name records the latent size.
* Every run logs `last_acc` and `best_acc`; best-over-training uses target labels for model
  selection, so report `last_acc` as the primary number.

## Run safety

A run directory is named from its arguments by `src/apps.py`, the same code the launcher
uses.  A trainer skips a run whose `done.json` exists, never deletes a finished run (only
`--overwrite` does), restarts an unfinished one, and holds a lock file with its PID so two
processes cannot run the same configuration.  Use `--tag` for protocol variants.
