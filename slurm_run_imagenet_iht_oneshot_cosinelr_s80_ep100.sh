#!/bin/bash
#SBATCH -J "imgnet"     # name of job in SLURM
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 20
#SBATCH -o joblog/slurm_run_imagenet_iht_oneshot_cosinelr_s80_ep100.txt
#SBATCH -e joblog/slurm_run_imagenet_iht_oneshot_cosinelr_s80_ep100.txt
#SBATCH --account=p490-24-t
#SBATCH --time=60
#SBATCH --mem=200GB
#or srun  -J "imgnet" -c 20 -p gpu -G 1 -o joblog/slurm_run_imagenet_iht_oneshot_cosinelr_s80_ep100.txt -e joblog/slurm_run_imagenet_iht_oneshot_cosinelr_s80_ep100.txt --account=p490-24-t --time=60 --mem=200GB slurm_run_imagenet_iht_oneshot_cosinelr_s80_ep100.sh

echo "Launched at $(date)"
echo "Job ID: ${SLURM_JOBID}"
echo "Node list: ${SLURM_NODELIST}"
echo "Submit dir.: ${SLURM_SUBMIT_DIR}"
echo "Numb. of cores: ${SLURM_CPUS_PER_TASK}"
echo $SHELL

echo "Loading cuda"
module load cuda/12.0.1

echo "Lets get this party started!"


declare -a gpu=0 #0,1,2,3
declare -a manual_seed=(3)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=imagenet \
	--dset_path=/projects/p490-24-t/data/imagenet/ \
	--arch=resnet50 \
	--config_path=./configs/neurips/iht_imagenet_resnet50_insta_cosinelr_s80_ep100.yaml \
	--workers=20 \
	--epochs=100 \
	--fp16 \
	--reset_momentum_after_recycling \
	--checkpoint_freq 10 \
	--batch_size=512 \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=iht_imagenet_resnet50_oneshot_cosinelr_fp16_s80_ep100 \
        --wandb_project "imagenet_resnet50" 

done

