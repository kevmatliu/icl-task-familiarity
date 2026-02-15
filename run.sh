#!/bin/bash
#SBATCH --job-name=icl_task_familiarity
#SBATCH --partition=kempner_requeue
#SBATCH --accoun=kempner_pehlevan_lab
#SBATCH --gres=gpu:4
#SBATCH --time==30:00

module load python/3.10.9-fasrc01

python main/experiments.py --task_id $SLURM_ARRAY_TASK_ID