#!/usr/bin/env bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --account=atmlaml
#SBATCH --cpus-per-task=4
#SBATCH --output=run-out_imagenet_w_LARS.%j
#SBATCH --error=run-err_imagenet_w_LARS.%j
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=develgpus
#SBATCH --job-name=finetune_s

source /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh

module load Python
pip list 
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python -m wandb offline


python pl_main_pretraining.py 