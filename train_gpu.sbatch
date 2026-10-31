#!/bin/bash
#SBATCH --job-name=vae-cxr
#SBATCH --partition=normal
#SBATCH --account=tritondroids
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/vae-cxr-%j.out
#SBATCH --error=outputs/vae-cxr-%j.err

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "Node: $(hostname)  Job: $SLURM_JOB_ID  Started: $(date)"
nvidia-smi || true

# Fail fast rather than silently training on CPU for hours
python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda available:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo 'ERROR: CUDA not available on this node (driver too old for the cu130 wheel?)'; exit 1; }

python main.py
echo "Finished: $(date)"
