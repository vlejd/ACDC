#!/bin/bash
#SBATCH -J "cifar10"     # name of job in SLURM
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -o joblog/slurm_run_cifar10_randomlabels_iht_no_da.txt
#SBATCH -e joblog/slurm_run_cifar10_randomlabels_iht_no_da.txt
#SBATCH --account=p490-24-t
#SBATCH --time=60
#SBATCH --mem=250GB
echo "Launched at $(date)"
echo "Job ID: ${SLURM_JOBID}"
echo "Node list: ${SLURM_NODELIST}"
echo "Submit dir.: ${SLURM_SUBMIT_DIR}"
echo "Numb. of cores: ${SLURM_CPUS_PER_TASK}"
echo $SHELL

echo "Loading cuda"
module load cuda/12.0.1

echo "Setting up conda"
module load lang
conda activate acdc_slurm


echo "Lets get this party started!"


declare -a gpu=0
declare -a manual_seed=(21)
declare -a num_rand=1000
#declare -a sparsities=(50 75 90 95)
declare -a sparsities=(50)

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#sparsities[@]};++i)); 
do
python main.py \
	--dset=cifar10 \
	--dset_path=/projects/p490-24-t/data/cifar10 \
	--arch=resnet20 \
	--config_path=./configs/neurips/iht_cifar10_resnet20_unstructured_insta_prune_freq20_${sparsities[i]}_constant.yaml \
	--workers=4 \
	--epochs=200 \
	--num_random_labels=${num_rand} \
	--batch_size=128 \
	--reset_momentum_after_recycling \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name="cifar10_random_${num_rand}_iht_oneshot_freq20_${sparsities[i]}_no_da" \
	--wandb_group="cifar10_resnet20_${num_rand}" \
	--wandb_name "iht_oneshot_freq20_sp${sparsities[i]}_no_da" \
        --wandb_project "acdc_cifar10_memorization" 

done
done
