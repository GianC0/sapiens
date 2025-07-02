#!/bin/bash

#Mahti
#SBATCH --account=project_2013416
#SBATCH --partition=gpusmall
## Specify number of GPUs per compute node, and NVME storage per node
#SBATCH --array=28
#SBATCH --gres=gpu:a100:2
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/scratch/project_2013416/sapiens/logs_%A_%a.out
module load pytorch
source /scratch/project_2013416/Lean/UMI/.venv/bin/activate
srun python optuna_objective.py --exp_id ${SLURM_ARRAY_TASK_ID}
