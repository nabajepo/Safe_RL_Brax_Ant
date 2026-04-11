#!/bin/bash
#SBATCH --account=def-cbelling-ab
#SBATCH --job-name=hard_constraint_pipeline
#SBATCH --output=hard_constraint_pipeline_%j.txt
#SBATCH --error=hard_constraint_pipeline_%j.txt
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

PROJECT_DIR="/home/jean17/projects/def-cbelling-ab/jean17/csi_4900_brax"
CONTAINER="${PROJECT_DIR}/python_3.10.sif"

MODEL_NAME="hard_constraint"
TIMESTEPS=1000000
RESULTS_ROOT="Results"
SEEDS=(0 1 2 3 4)

STEPS_PER_ENV=256
PPO_EPOCHS=8
MINIBATCH_SIZE=4096
HIDDEN_DIM=256

LEARNING_RATE=3e-4
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_EPS=0.2
VALUE_COEF=0.5
ENTROPY_COEF=0.01
MAX_GRAD_NORM=1.0

EVAL_EVERY_ITERS=5
EVAL_EPS=50
FINAL_EVAL_EPS=200
ROLLOUTS_PER_SEED=3
CHECKPOINT_EVERY_TIMESTEPS=200000

NUM_ENVS=512
MAX_STEPS=300

BUDGET_TAG="t$((TIMESTEPS / 1000000))m"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "Submitting CSI4900 pipeline for model: ${MODEL_NAME}"
echo "Project directory : ${PROJECT_DIR}"
echo "Container         : ${CONTAINER}"
echo "Budget tag        : ${BUDGET_TAG}"
echo "Seeds             : ${SEEDS[*]}"
echo "Submit job ID     : ${SLURM_JOB_ID}"
echo "============================================================"

TRAIN_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --account=def-cbelling-ab
#SBATCH --job-name=${MODEL_NAME}_s${SEED}
#SBATCH --output=${MODEL_NAME}_seed_${SEED}_%j.txt
#SBATCH --error=${MODEL_NAME}_seed_${SEED}_%j.txt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR}"
CONTAINER="${CONTAINER}"

cd "\${PROJECT_DIR}"

echo "============================================================"
echo "Starting training job for model=${MODEL_NAME}, seed=${SEED}"
echo "Project directory : \${PROJECT_DIR}"
echo "Container         : \${CONTAINER}"
echo "Job ID            : \${SLURM_JOB_ID}"
echo "Node              : \${SLURMD_NODENAME}"
echo "============================================================"

module load apptainer/1.4.5

apptainer exec \
  --cleanenv \
  --nv \
  --bind "\${PROJECT_DIR}:\${PROJECT_DIR}" \
  "\${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '\${PROJECT_DIR}'

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

    echo 'Starting training for model=${MODEL_NAME}, seed=${SEED} ...'
    python train_model.py \
      --model_name ${MODEL_NAME} \
      --seed ${SEED} \
      --timesteps ${TIMESTEPS} \
      --results_root ${RESULTS_ROOT} \
      --steps_per_env ${STEPS_PER_ENV} \
      --ppo_epochs ${PPO_EPOCHS} \
      --minibatch_size ${MINIBATCH_SIZE} \
      --hidden_dim ${HIDDEN_DIM} \
      --learning_rate ${LEARNING_RATE} \
      --gamma ${GAMMA} \
      --gae_lambda ${GAE_LAMBDA} \
      --clip_eps ${CLIP_EPS} \
      --value_coef ${VALUE_COEF} \
      --entropy_coef ${ENTROPY_COEF} \
      --max_grad_norm ${MAX_GRAD_NORM} \
      --eval_every_iters ${EVAL_EVERY_ITERS} \
      --eval_eps ${EVAL_EPS} \
      --final_eval_eps ${FINAL_EVAL_EPS} \
      --rollouts_per_seed ${ROLLOUTS_PER_SEED} \
      --checkpoint_every_timesteps ${CHECKPOINT_EVERY_TIMESTEPS} \
      --num_envs ${NUM_ENVS} \
      --max_steps ${MAX_STEPS}
  "
EOF
)
    TRAIN_JOB_IDS+=("${JOB_ID}")
    echo "Submitted training job for seed ${SEED}: ${JOB_ID}"
done

DEPENDENCY=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")

AGG_JOB_ID=$(sbatch --parsable --dependency=afterok:${DEPENDENCY} <<EOF
#!/bin/bash
#SBATCH --account=def-cbelling-ab
#SBATCH --job-name=${MODEL_NAME}_aggregate
#SBATCH --output=${MODEL_NAME}_aggregate_%j.txt
#SBATCH --error=${MODEL_NAME}_aggregate_%j.txt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR}"
CONTAINER="${CONTAINER}"
MODEL_NAME="${MODEL_NAME}"
TIMESTEPS="${TIMESTEPS}"
RESULTS_ROOT="${RESULTS_ROOT}"
SEEDS_STR="${SEEDS[*]}"
BUDGET_TAG="${BUDGET_TAG}"

cd "\${PROJECT_DIR}"

module load apptainer/1.4.5

apptainer exec \
  --cleanenv \
  --bind "\${PROJECT_DIR}:\${PROJECT_DIR}" \
  "\${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '\${PROJECT_DIR}'

    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

    source .venv/bin/activate

    echo 'Aggregating model=\${MODEL_NAME} ...'
    python aggregate_models.py \
      --model_name \${MODEL_NAME} \
      --seeds \${SEEDS_STR} \
      --timesteps \${TIMESTEPS} \
      --results_root \${RESULTS_ROOT}

    echo 'Sending completion email ...'
    MAIL_USER='nabajepo@gmail.com' \
    MAIL_PASS='CHANGE_ME_APP_PASSWORD' \
    MAIL_TO='jnahi100@uottawa.ca' \
    MAIL_SUBJECT='CSI 4900 - hard_constraint terminé' \
    MAIL_BODY=\$'Salut,\\n\\nLe pipeline du modèle hard_constraint est terminé sur Compute Canada.\\n\\nRésultats :\\n\${PROJECT_DIR}/\${RESULTS_ROOT}/\${BUDGET_TAG}/\${MODEL_NAME}\\n\\nMessage automatique.' \
    python mail_me.py
  "
EOF
)

echo "Submitted aggregation job: ${AGG_JOB_ID}"

echo "============================================================"
echo "Pipeline submission completed for model: ${MODEL_NAME}"
echo "Training jobs : ${TRAIN_JOB_IDS[*]}"
echo "Aggregate job : ${AGG_JOB_ID}"
echo "Results root  : ${PROJECT_DIR}/${RESULTS_ROOT}"
echo "============================================================"