#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=hai_consultantfzj
#SBATCH --cpus-per-task=4
#SBATCH --output=run-out_mmasking.%j
#SBATCH --error=run-err_mmasking.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --job-name=masking

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh

module load Python

python -m wandb offline

module list

python main_pretraining_masking.py 