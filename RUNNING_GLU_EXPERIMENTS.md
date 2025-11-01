# Running GLU Experiments - Quick Guide

## ✅ Script Updated!

The `run_glu_experiments.py` script now uses **Chinchilla-optimal training steps by default**.

### What Changed:
- `--steps` argument now defaults to `None` (automatic Chinchilla calculation)
- When `--steps` is not specified, the code automatically calculates: `20 tokens per parameter`
- You can still override with `--steps N` for quick testing

---

## 📊 Expected Training Steps (Automatic Mode)

| Model | Parameters | Batch×Seq | Chinchilla Steps | Tokens | Est. Time (RTX 4090) |
|-------|-----------|-----------|------------------|--------|---------------------|
| chinchilla-44m | 47.3M | 192×512 | ~9,626 | 946M | ~2-3 hours |
| chinchilla-74m | 79M | 192×512 | ~16,065 | 1.58B | ~4-5 hours |
| chinchilla-251m | 251M | 192×512 | ~51,065 | 5.02B | ~12-15 hours |

**Formula**: `steps = (20 × num_parameters) / (batch_size × sequence_length)`

---

## 🚀 Recommended Commands

### 1. Quick Test (10 steps, verify setup)
```bash
python run_glu_experiments.py \
    --model_size chinchilla-44m \
    --steps 10 \
    --warmup_steps 100 \
    --inner_size_multiple_of 64 \
    --no_neptune \
    --variants baseline
```
**Time**: ~2 minutes  
**Purpose**: Verify data loads, model builds, training works

---

### 2. Full Baseline + SwiGLU Comparison (Recommended)
```bash
python run_glu_experiments.py \
    --model_size chinchilla-44m \
    --warmup_steps 100 \
    --inner_size_multiple_of 64 \
    --variants baseline swiglu \
    --no_neptune
```
**Experiments**: 6 (2 variants × 3 LRs)  
**Steps per experiment**: ~9,626 (Chinchilla-optimal)  
**Total time**: ~12-18 hours  
**Purpose**: Compare best non-GLU vs best GLU (as your collaborator suggested)

---

### 3. Full Comparison (All Variants)
```bash
python run_glu_experiments.py \
    --model_size chinchilla-44m \
    --warmup_steps 100 \
    --inner_size_multiple_of 64 \
    --no_neptune
```
**Experiments**: 9 (3 variants × 3 LRs)  
**Steps per experiment**: ~9,626 (Chinchilla-optimal)  
**Total time**: ~18-27 hours  
**Purpose**: Full comparison including GeGLU

---

### 4. Custom Learning Rate Sweep
```bash
python run_glu_experiments.py \
    --model_size chinchilla-44m \
    --warmup_steps 100 \
    --inner_size_multiple_of 64 \
    --lr_sweep 0.0015 0.001 \
    --variants baseline swiglu \
    --no_neptune
```
**Experiments**: 4 (2 variants × 2 LRs)  
**Purpose**: Test specific learning rates

---

### 5. Scale to Larger Models (After finding best LR on 44m)
```bash
# After finding best LR from 44m experiments, run on 74m
python run_glu_experiments.py \
    --model_size chinchilla-74m \
    --warmup_steps 100 \
    --inner_size_multiple_of 64 \
    --lr_sweep 0.0015 \
    --variants baseline swiglu \
    --no_neptune
```
**Purpose**: Validate findings on larger model

---

## 📝 Understanding the Output

### During Training:
```
GLU Experiment Configuration
================================================================================
Model size: chinchilla-44m
Training steps per experiment: Chinchilla-optimal (auto, ~20 tokens per parameter)
Learning rates: [0.0015, 0.001, 0.0007]
Warmup steps: 100
inner_size_multiple_of: 64
Variants: ['baseline', 'swiglu']
Neptune logging: False
================================================================================

Generated 6 experiment configurations

[1/6] Running: c44m_baseline_lr2e-3_w100
...
Using Chinchilla-optimal number of steps: 9626
...
```

### Key Metrics to Watch:
- **Initial loss**: Should be ~11.5 (random initialization)
- **Final validation loss**: Lower is better
- **Loss curve**: Should decrease smoothly
- **Memory usage**: Should stay under 24GB (RTX 4090)

---

## 🎯 What to Expect

### Parameter Counts (with inner_size_multiple_of=64):
- **Baseline**: 47,307,008 params
- **SwiGLU**: 47,307,008 params (±1% due to rounding)
- **GeGLU**: 47,307,008 params (±1% due to rounding)

This ensures **fair comparison** as intended by the GLU paper.

### Expected Results (based on literature):
- **SwiGLU** should achieve **~2-5% lower loss** than baseline at same compute
- **GeGLU** should be slightly worse than SwiGLU
- Best learning rate might differ between variants

---

## 🔍 Monitoring Progress

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Tail the logs (if redirected):
```bash
tail -f experiments/zheren/glu-exps/experiment.log
```

### Check intermediate results:
The script prints validation loss every 100 steps by default.

---

## ⚠️ Important Notes

1. **Validation data**: Make sure you have `data/slimpajama_validation/` downloaded
   - If missing, download with: `python -c "from download_data import download_slimpajama; download_slimpajama(split='validation', num_files=10)"`

2. **Disk space**: Each experiment checkpoint can be large. Monitor disk usage.

3. **Interrupting**: You can Ctrl+C to stop. Completed experiments are saved.

4. **Neptune logging**: Remove `--no_neptune` if you want to log to Neptune for tracking.

---

## 📊 Analyzing Results

After experiments complete, compare:
1. **Final validation loss** for each variant
2. **Loss at same number of tokens** (all should see ~946M tokens)
3. **Training stability** (smooth loss curves vs spiky)
4. **Best learning rate** for each variant

Your collaborator mentioned:
> "Should compare models by loss achieved for given compute"

Since all experiments run for the same number of steps with the same batch size, they all use the **same compute budget** - perfect for fair comparison!

---

## 🚦 Next Steps After Results

1. **Identify best LR** for baseline and SwiGLU
2. **Compare best baseline vs best SwiGLU** at matched compute
3. **If SwiGLU wins**: Scale to 74m and 251m with best settings
4. **If baseline wins**: Investigate (might indicate implementation issue or hyperparameter mismatch)
5. **Share results** with your collaborator

---

## 💡 Pro Tips

- Start with **Option 2** (baseline + swiglu only) for faster iteration
- Run **Option 1** first to verify everything works
- Use `screen` or `tmux` for long-running experiments:
  ```bash
  screen -S glu_experiments
  python run_glu_experiments.py --model_size chinchilla-44m --variants baseline swiglu --no_neptune
  # Ctrl+A, D to detach
  # screen -r glu_experiments to reattach
  ```

