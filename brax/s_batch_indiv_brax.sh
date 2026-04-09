#!/bin/bash
#SBATCH --account=def-cbelling-ab
#SBATCH --job-name=csi4900_soft_constraint
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="/home/jean17/projects/def-cbelling-ab/jean17/csi_4900_brax"
CONTAINER="${PROJECT_DIR}/python_3.10.sif"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "Starting CSI4900 Brax individual job"
echo "Project directory : ${PROJECT_DIR}"
echo "Container         : ${CONTAINER}"
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : ${SLURMD_NODENAME}"
echo "============================================================"

module load apptainer/1.4.5

apptainer exec \
    --cleanenv \
    --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${CONTAINER}" \
    bash -lc "
        set -euo pipefail
        cd '${PROJECT_DIR}'

        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
        export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
        export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

        source .venv/bin/activate

        echo 'Python:' \$(python --version)
        echo 'Checking imports...'
        python -c 'import jax, brax, optax, numpy, pandas, matplotlib; print(\"All imports OK\")'

        echo 'JAX devices:'
        python -c 'import jax; print(jax.devices())'

        echo 'Starting training...'
        python train_individual_brax_env.py \
          --model soft_constraint \
          --timesteps 500000 \
          --seeds 0 1 2 \
          --results_root Results_easy_cfg \
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
          --eval_every_iters 5 \
          --eval_eps 50 \
          --final_eval_eps 200 \
          --rollouts_per_seed 3 \
          --num_envs 16 \
          --max_steps 300
    "

echo "============================================================"
echo "Job completed"
echo "Results should be in:"
echo "${PROJECT_DIR}/Results_easy_cfg"
echo "============================================================"