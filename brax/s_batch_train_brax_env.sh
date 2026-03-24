#!/bin/bash
#SBATCH --account=def-cbelling-ab
#SBATCH --job-name=csi4900_brax
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="/home/jean17/projects/def-cbelling-ab/jean17/csi_4900_brax"
CONTAINER="${PROJECT_DIR}/python_3.10.sif"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "Starting CSI4900 Brax job"
echo "Project directory : ${PROJECT_DIR}"
echo "Container         : ${CONTAINER}"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : ${SLURMD_NODENAME}"
echo "============================================================"

module load apptainer

apptainer exec \
    --cleanenv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${CONTAINER}" \
    bash -lc "
        set -euo pipefail
        cd '${PROJECT_DIR}'

        source .venv/bin/activate

        echo 'Python:' \$(python --version)
        echo 'Checking imports...'
        python -c 'import jax, brax, optax, numpy, pandas, matplotlib; print(\"All imports OK\")'

        echo 'Starting training...'
        python train_brax_env.py \
          --timesteps 200000 \
          --seeds 0 1 2 3 4 \
          --results_root Results \
          --steps_per_env 512 \
          --ppo_epochs 8 \
          --minibatch_size 1024 \
          --hidden_dim 256 \
          --learning_rate 3e-4 \
          --gamma 0.99 \
          --gae_lambda 0.95 \
          --clip_eps 0.2 \
          --value_coef 0.5 \
          --entropy_coef 0.01 \
          --max_grad_norm 1.0 \
          --eval_every_iters 2 \
          --eval_eps 50 \
          --final_eval_eps 200 \
          --rollouts_per_seed 3 \
          --num_envs 16 \
          --max_steps 300
    "

echo "============================================================"
echo "Job completed"
echo "Results should be in:"
echo "${PROJECT_DIR}/Results"
echo "============================================================"