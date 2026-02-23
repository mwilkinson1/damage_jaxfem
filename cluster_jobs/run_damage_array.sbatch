#!/bin/bash
#SBATCH -A abuganza
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=damage_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-11%12

set -euo pipefail

# ---- EDIT THIS IF YOUR IMAGE IS ELSEWHERE ----
IMAGE="${SCRATCH}/containers/jax_fem/jax_fem.sif"
# --------------------------------------------

CODE_DIR="${SLURM_SUBMIT_DIR}"
RUN_ROOT="${SCRATCH}/coupled_damage_runs"

mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

# Keep Apptainer cache off $HOME
export APPTAINER_CACHEDIR="${SCRATCH}/.apptainer_cache"
mkdir -p "${APPTAINER_CACHEDIR}"

meshes=(
  "notched_plate_C3D20_coarse.inp"
  "notched_plate_C3D20_medium.inp"
  "notched_plate_C3D20_fine.inp"
)
steps_list=(26 51 101 201)

mesh_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))
step_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))

mesh_file="${meshes[$mesh_idx]}"
nsteps="${steps_list[$step_idx]}"

mesh_path="${CODE_DIR}/${mesh_file}"

if [[ ! -f "${CODE_DIR}/main.py" ]]; then
  echo "ERROR: main.py not found in ${CODE_DIR}"
  exit 1
fi
if [[ ! -f "${mesh_path}" ]]; then
  echo "ERROR: mesh not found: ${mesh_path}"
  exit 1
fi
if [[ ! -f "${IMAGE}" ]]; then
  echo "ERROR: apptainer image not found: ${IMAGE}"
  exit 1
fi

mesh_stem="$(basename "${mesh_file}" .inp)"

# Unique run directory per (mesh, steps) so outputs never overwrite
RUN_DIR="${RUN_ROOT}/${mesh_stem}/steps_${nsteps}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

echo "=== Running ==="
echo "Task: ${SLURM_ARRAY_TASK_ID}  Mesh: ${mesh_file}  Steps: ${nsteps}"
echo "Run dir: ${RUN_DIR}"
echo "==============="

apptainer exec "${IMAGE}" \
  python "${CODE_DIR}/main.py" \
    --mesh "${mesh_path}" \
    --steps "${nsteps}" \
    --out-dir "${RUN_DIR}"