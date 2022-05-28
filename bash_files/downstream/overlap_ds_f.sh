#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=atmlaml
#SBATCH --cpus-per-task=4
#SBATCH --output=out_o.%j
#SBATCH --error=err_o.%j
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpus
#SBATCH --job-name=l_abc

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh
module load Python

python -m wandb offline

module list

python /p/project/hai_ssl4sc/hai_thesis_simcLR/orig_wo_normalize_intransform/down_stream_simCLR.py \
    --num_classes 10 \
    --epochs 50 \
    --batch_size 30\
    --encoder_fine_tune \
    --ckpt /p/project/hai_ssl4sc/hai_thesis_simcLR/orig_wo_normalize_intransform/bash_files/ckpt_all/baseline_wo_norm_ckpt/ckpt_10000__final_overlap.pt