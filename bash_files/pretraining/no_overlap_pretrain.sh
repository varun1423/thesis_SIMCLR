#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=hai_ssl4sc
#SBATCH --cpus-per-task=4
#SBATCH --output=run-err_no_o_wo_norm_.%j
#SBATCH --error=run-err_no_o_wo_norm_.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --job-name=bas

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh
module load Python

python -m wandb offline

module list
 
python /p/project/hai_ssl4sc/hai_thesis_simcLR/orig_wo_normalize_intransform/main_pretraining_iterloader.py \
    --batch_size 300 \
    --steps 20000 \
    --base_encoder ResNet18 \
    --cropping_strategy crops_no_overlap_only \
    --image_input_channel 3\
    --npy_file orig_images_200_iter.npy \
    --proj_out_feats 256 \
    --proj_hidden_feats 2048 \
    --crop_size 224 \
    --temperature 0.3 \
    --comment baseline_without_tensor_norm \
    --wandb_project baseline_without_tensor_norm