#!/bin/bash
#SBATCH --job-name=versa_quality
#SBATCH --output=logs/versa_%j.out
#SBATCH --error=logs/versa_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00 


set -euo pipefail

# ---- 1. Environment setup (don't trust .bashrc to run in batch) ----
module load miniconda          # whatever you use interactively
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate versa

# ---- 2. Re-export everything that lives in your activation hook ----
# In theory `conda activate` runs them. In practice batch shells sometimes
# skip activation hooks. Belt-and-suspenders:
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export OMP_NUM_THREADS=1
export ORT_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ---- 3. Sanity checks BEFORE the real work ----
# Catch broken environments early instead of after 6 hours of queue.
which python
python --version
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible'; print(torch.__version__, torch.version.cuda)"
python -c "from versa.bin.scorer import main; print('versa import ok')"

cd /home/prudhome/audio_quality

# ---- 4. The actual run ----
# Single dataset:
python run_versa_quality.py \
    --csv          /home/prudhome/audio_quality/test_short.csv \
    --dataset_name test_enenlhet \
    --out_dir      /scratch/prudhome/results/test_enenlhet/ \
    --versa_repo   /home/prudhome/versa \
    --score_config /home/prudhome/audio_quality/audio_quality_config.yaml \
    --has_header \
    --no_text \
    --use_gpu

echo "Job complete."
