#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=atmlaml
#SBATCH --cpus-per-task=4
#SBATCH --output=out_mul.%j
#SBATCH --error=err_mul.%j
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=develgpus
#SBATCH --job-name=l_abc

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh
module load Python

python -m wandb offline

module list

python /p/project/hai_ssl4sc/hai_thesis_simcLR/orig_wo_normalize_intransform/down_stream_simCLR.py \
    --num_classes 10 \
    --epochs 50 \
    --batch_size 500\
    --ckpt /p/project/hai_ssl4sc/hai_thesis_simcLR/orig_wo_normalize_intransform/bash_files/simclr_alll_crops_baseline/ckpt_all/ckpt_5000__final_multi_5k.pt