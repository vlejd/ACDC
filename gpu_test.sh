#!/bin/bash
#SBATCH -J "sample_cifar10"     # name of job in SLURM
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -o joblog/run_cifar10_randomlabels_iht_no_da.txt
#SBATCH -e joblog/run_cifar10_randomlabels_iht_no_da.txt
#SBATCH --account=p490-24-t
#SBATCH --time=10

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
# conda env list
# conda init bash
conda activate acdc_slurm

# pip freeze


echo "Lets get this party started!"
bash run_cifar10_randomlabels_iht_no_da.sh


