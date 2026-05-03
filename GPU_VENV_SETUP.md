# GPU-Specific Virtual Environments

This project supports GPU-specific virtual environments to avoid rebuilding vLLM when switching between different GPU types.

## Overview

- **A100**: `sm_80`, `sm_81` → `.venv-a100`
- **H100**: `sm_90` → `.venv-h100`
- **H200**: `sm_90a` → `.venv-h200`

`run_project.sh` automatically detects your GPU type and uses the corresponding venv.

## Setup

### Option 1: Build All Three (Recommended for Multi-GPU Clusters)

If you have access to different GPU types and want to pre-build venvs for all of them:

```bash
# Build A100 venv
VENV_DIR=.venv-a100 bash setup_env.sh

# Build H100 venv
VENV_DIR=.venv-h100 bash setup_env.sh

# Build H200 venv (if available)
VENV_DIR=.venv-h200 bash setup_env.sh
```

This takes time but you only need to do it once. After that, switching GPUs is instant—no rebuild!

### Option 2: Build Only What You Need

If you only use one or two GPU types, just build those:

```bash
# Only A100
VENV_DIR=.venv-a100 bash setup_env.sh

# Or only H100
VENV_DIR=.venv-h100 bash setup_env.sh
```

If you later switch to a GPU type without a pre-built venv, `run_project.sh` will exit with a clear error message telling you how to build it.

### Option 3: Use a Custom venv Path

If you prefer a specific venv location, set `VENV_DIR` explicitly:

```bash
VENV_DIR=/my/custom/path bash setup_env.sh
VENV_DIR=/my/custom/path bash run_project.sh
```

## How It Works

1. **Detection**: `run_project.sh` runs `torch.cuda.get_device_capability()` to detect your GPU's compute capability (e.g., `sm_80` for A100).

2. **Mapping**: The capability is mapped to a short GPU type name:
   - `80`, `81`, `89` → `a100`
   - `90` → `h100`
   - `90a` → `h200`

3. **Venv Selection**: Uses `.venv-{gpu_type}` from the repo root (e.g., `.venv-h100`).

4. **Activation**: Activates the selected venv and runs your experiments.

## Migration from Old Single venv

If you have an existing `.venv` that was built for a specific GPU:

```bash
# Move it to the GPU-specific name
mv .venv .venv-a100  # if it was built on A100
# or
mv .venv .venv-h100  # if it was built on H100
```

Next time you run `run_project.sh` on that GPU, it will find the corresponding venv automatically.

## Troubleshooting

### "ERROR: venv not found at .venv-X100"

You need to build the venv for that GPU type:

```bash
VENV_DIR=.venv-h100 bash setup_env.sh
```

(Replace `h100` with the GPU type you're using.)

### How do I know which GPU I have?

Run `nvidia-smi` and check the GPU model, or let `run_project.sh` detect it:

```bash
bash run_project.sh --dry-run
```

This will show the detected GPU capability without running experiments.

### Can I use a single venv across multiple GPU types?

Not recommended—vLLM's CUDA kernels are compiled for specific GPU architectures. A single venv may fail on a different GPU with `cudaErrorNoKernelImageForDevice`.

The whole point of GPU-specific venvs is to avoid this problem!

## Example: Typical Multi-GPU Cluster Workflow

```bash
# First time only: build venvs for all available GPU types
VENV_DIR=.venv-a100 bash setup_env.sh  # ~30–45 min
VENV_DIR=.venv-h100 bash setup_env.sh  # ~30–45 min

# Now, run experiments on any GPU type—instant venv selection
bash run_project.sh --profile smoke                # auto-detects GPU
bash run_project.sh --profile medium --gpus 0,1   # auto-detects GPU, uses TP=2
# ... schedule another job on a different GPU type ...
bash run_project.sh --profile full --gpus 0       # auto-detects, reuses H100 venv (or H200 if that's what's available)
```

No rebuilds needed! ✓
