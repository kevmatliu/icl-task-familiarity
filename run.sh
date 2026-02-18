#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=32G
#SBATCH --job-name=icl_task_familiarity
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# module load cuda
module load python/3.10.9-fasrc01
source ~/icl-task-familiarity/icl-task-familiarity/venv/bin/activate

python main/experiments.py