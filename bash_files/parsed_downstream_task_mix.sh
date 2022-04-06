#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=hai_ssl4sc
#SBATCH --cpus-per-task=4
#SBATCH --output=run-out_bias__masking_linear_img.%j
#SBATCH --error=run-err_bias__masking_linear_img.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --job-name=l_mix_crops

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh
module load Python

python -m wandb offline

module list

python /p/project/hai_consultantfzj/set_up/simclr_with_down_stream/thesis_SIMCLR/down_stream_simCLR.py \
    --num_classes 10 \
    --epochs 50 \
    --batch_size 320\
    --ckpt /p/project/hai_consultantfzj/set_up/simclr_with_down_stream/thesis_SIMCLR/final_split_ckpt/first_run_on_final_data/ckpt_9999_fpnarm1v_mix_crops_.pt
