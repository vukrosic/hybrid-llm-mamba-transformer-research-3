## Hybrid Transformer–Mamba Experiments (W&B Export) — Report

- **Date**: 2025-08-21
- **Project**: transformer-mamba-llm-research
- **Source scripts**: `experimental_training.py`, `train_hybrid_llm.py`, `shared_data.py`
- **Results file**: `wandb_experiment_results.csv`
- **Launcher**: `run_experiments.sh`

### What these experiments test
This study evaluates hybrid LLM blocks that interleave Attention (A) and SSM/Mamba (M) layers. Each experiment sets a fixed layer pattern (e.g., `AMAMAMAM`) and trains a ~24–26M parameter model on the SmolLM Cosmopedia-v2 corpus, comparing validation loss and perplexity across architectural patterns.

- **A**: Transformer-style self-attention
- **M**: SSM/Mamba-style mixer (in `experimental_training.py`, M layers are replaced with an improved SSM: `ImprovedSSM`)

### Interpreting W&B “final” vs non-final columns
From `experimental_training.py`:
- Periodic logging during training:
  - `train_loss` logged every `log_every` steps.
  - `val_loss`, `val_perplexity` logged every `eval_every` steps, computed over a subset of validation batches (`num_eval_batches`, default 50).
  - `best_val_loss` tracked as the best periodic validation loss seen so far.
- Final evaluation at the end:
  - Runs a full validation pass over the entire validation loader (not just a subset).
  - Logs to W&B a results dict that includes `final_val_loss`, `final_val_perplexity`, and `best_val_loss`.

Therefore in the W&B export:
- `final_val_loss` and `final_val_perplexity` are computed on the full validation set at the very end of training and explicitly logged once.
- `val_loss` and `val_perplexity` are the last periodically evaluated values during training (subset evaluation).
- `best_val_loss` is the minimum periodic validation loss observed at any evaluation point.
- Small gaps between `final_val_*` and the last `val_*` are expected (full-set evaluation vs subset evaluation, and possibly an early-stopped or later checkpoint).

### Experimental setup

- **Data and tokenization** (`shared_data.py`):
  - Tokenizer: `HuggingFaceTB/SmolLM-135M` (pad token set to pad or EOS).
  - Dataset: `HuggingFaceTB/smollm-corpus`, split `train`, config `cosmopedia-v2`, streamed.
  - Documents: typically 50,000 (see `ExperimentConfig.num_documents`), tokenized to length ~4k chars per doc.
  - Split: 85% train / 15% val, then flattened to token streams.
  - Sequence length: 512.
  - Train dataset uses overlapping stride at ~0.8× sequence length; val uses non-overlapping stride.

- **Model** (`train_hybrid_llm.py`, extended by `experimental_training.py`):
  - Hidden size: 384; heads: 8; SSM state size: 16; dropout: 0.1.
  - Weight tying between embedding and LM head.
  - Hybrid blocks: per-layer mixer is chosen by pattern char (`A` = Attention, `M` = SSM). During experiments, SSM mixers are upgraded to `ImprovedSSM` for stability.

- **Training** (`experimental_training.py` defaults via `ExperimentConfig`):
  - Steps: 10,000; batch size: 32; grad accumulation: 1 (effective batch size 32).
  - Optimizer: AdamW (betas 0.9/0.95, weight decay 0.01, eps 1e-8).
  - LR schedule: cosine with 1,000 warmup steps; base LR 4e-4.
  - Mixed precision: AMP with `GradScaler`.
  - Gradient clipping: 1.0.
  - Evaluation: every 200 steps on a subset; final pass evaluates the full val set.
  - Early stopping: patience 10 on periodic `val_loss`.

- **Launch orchestration** (`run_experiments.sh`):
  - Runs a suite of named patterns across 4 phases (baselines, alternating, structured, research-inspired).
  - Uses `--use_wandb` to log into project `hybrid-patterns`.

### Results (from `wandb_experiment_results.csv`)
Sorted by lowest `final_val_perplexity` (lower is better).

| Name | Layers | Pattern | Runtime (s) | Num Params (M) | Best Val Loss | Final Val Loss | Final Val PPL |
|---|---:|---|---:|---:|---:|---:|---:|
| double_alternate_10L | 10 | AAMMAAMMAA | 827 | 26.1 | 3.4828 | 3.4900 | 32.79 |
| fibonacci_9L | 9 | MAMMAMMMA | 981 | 26.2 | 3.4971 | 3.5043 | 33.26 |
| alternate_A_first_8L | 8 | AMAMAMAM | 810 | 25.0 | 3.5046 | 3.5101 | 33.45 |
| alternate_M_first_8L | 8 | MAMAMAMA | 797 | 25.0 | 3.5100 | 3.5142 | 33.59 |
| sandwich_8L | 8 | AMMMMMMA | 966 | 25.6 | 3.5396 | 3.5462 | 34.68 |
| thick_sandwich_8L | 8 | AAMMMMAA | 796 | 25.0 | 3.5426 | 3.5486 | 34.76 |
| unet_style_7L | 7 | AMMMMMA | 863 | 24.7 | 3.5697 | 3.5762 | 35.74 |
| grouped_M4A4 | 8 | MMMMAAAA | 794 | 25.0 | 3.5708 | 3.5765 | 35.75 |
| increasing_attention_7L | 7 | MMMMAAA | 777 | 24.4 | 3.5988 | 3.6051 | 36.79 |
| grouped_A4M4 | 8 | AAAAMMMM | 795 | 25.0 | 3.6426 | 3.6501 | 38.48 |
| baseline_mamba_8L | 8 | MMMMMMMM | 1161 | 26.3 | 3.7530 | 3.7634 | 43.09 |
| baseline_attention_8L | 8 | AAAAAAAA | 570 | 23.6 | 3.9722 | 3.9761 | 53.31 |

Notes:
- Models are ~23.6M–26.3M parameters; GPU memory usage peaks around ~9.3 GB in these runs.
- Periodic `val_perplexity` in the CSV is close to the final, but final is preferred for fair comparison (full validation set).

### Key findings
- **Best overall**: double alternating 10-layer pattern `AAMMAAMMAA` with the lowest final perplexity (≈32.79).
- **Strong performers**: Fibonacci-inspired `MAMMAMMMA`, and simple alternations (`AMAMAMAM`, `MAMAMAMA`).
- **Sandwich designs** (`AMMMMMMA`, `AAMMMMAA`) are competitive but a step behind the best alternations in these runs.
- **Baselines**: pure Mamba outperforms pure Attention, but both trail hybrid designs.

### Interpreting small metric differences
- The script’s periodic validation uses a subset of batches, whereas the final evaluation covers the entire validation set. This can yield slight shifts between `val_*` and `final_val_*`.
- `best_val_loss` records the best subset-validation checkpoint. Final performance reflects the model state at training end (or early stopping), evaluated on the full val set.

### How to reproduce

1) Run the full suite (logs go to `logs/`, artifacts to `experiments/`):
```bash
bash run_experiments.sh
```

2) Run a single pattern with W&B logging:
```bash
python experimental_training.py \
  --pattern "AAMMAAMMAA" \
  --name "double_alternate_10L" \
  --steps 10000 \
  --use_wandb
```

3) Force data retokenization (clears cache) if needed:
```bash
python experimental_training.py --pattern "AMAMAMAM" --use_wandb --force_reload_data
```

Requirements:
- Dataset access and internet for streaming (`datasets`).
- W&B credentials configured (e.g., `WANDB_API_KEY`) if logging is enabled.

### Recommendations
- Focus further training on the top 2–3 patterns (e.g., `AAMMAAMMAA`, `MAMMAMMMA`, `AMAMAMAM`) with longer schedules and/or larger models.
- Consider ablations on attention/SSM ratio and layer order within the best motifs.
- Report final metrics from full validation runs (as already captured by `final_val_*`), not last periodic values.

### Appendix: implementation details
- SSM stability improvements in `ImprovedSSM`: clamped softplus deltas, constrained A, layer norm after convolution, explicit decay clamping, and SiLU gates.
- Training loop uses AMP + gradient clipping + AdamW + cosine schedule with warmup, and early stopping on periodic `val_loss`.
- Data split is doc-based (85/15), then flattened into sequences with padding and stride.


