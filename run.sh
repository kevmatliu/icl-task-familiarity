#!/bin/bash
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mem=32G
#SBATCH --array=0-15
#SBATCH --job-name=icl_task_familiarity
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --ntasks-per-core=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00

# module load cuda
module load python/3.10.9-fasrc01
source ~/icl-task-familiarity/icl-task-familiarity/venv/bin/activate

python main/experiments.py