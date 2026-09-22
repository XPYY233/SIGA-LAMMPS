#!/bin/bash
# SIGA-LAMMPS HPC smoke test job.
#
# Validates: SLURM submission, module load, lmp execution, log collection, and
# whether /scratch exists on a COMPUTE node (it does not on the login node).
#
# Resource ceilings follow config/.env: 1 node, few tasks, short walltime.

#SBATCH --job-name=siga-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

echo "=== ENVIRONMENT ==="
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"

echo
echo "=== SCRATCH PROBE (compute node) ==="
echo "SCRATCH_VAR=${SCRATCH:-<unset>}"
if [ -d /scratch ]; then
  echo "SCRATCH_MOUNTED=yes"
  ls -ld /scratch 2>&1
else
  echo "SCRATCH_MOUNTED=no"
fi

echo
echo "=== MODULE LOAD ==="
if ! command -v module >/dev/null 2>&1; then
  # `module` is a shell function from the module init script.
  if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
fi
module load lammps/20250722-gcc-12.3.0-openmpi
echo "MODULE_STATUS=$?"
echo "LMP_PATH=$(command -v lmp)"

echo
echo "=== LAMMPS RUN ==="
if [ ! -f in.smoke ]; then
  echo "FATAL: in.smoke not found in $(pwd)"
  exit 2
fi
lmp -in in.smoke -log log.lammps
LMP_EXIT=$?
echo "LMP_EXIT=${LMP_EXIT}"

echo
echo "=== ARTIFACTS ==="
ls -la

if [ "${LMP_EXIT}" -ne 0 ]; then
  echo "=== LAMMPS ERROR TAIL ==="
  tail -n 30 log.lammps 2>/dev/null
  exit "${LMP_EXIT}"
fi

echo "=== THERMO TABLE ==="
sed -n '/Step/,/Loop time/p' log.lammps 2>/dev/null | tail -n 20
echo "SMOKE_TEST=PASS"
