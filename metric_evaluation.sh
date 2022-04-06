#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=hai_ssl4sc
#SBATCH --cpus-per-task=4
#SBATCH --output=run-out_itr_.%j
#SBATCH --error=run-err_itr_.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --job-name=wo_rot

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh
module load Python

python -m wandb offline

module list

python evaluation_and_metrics.py 