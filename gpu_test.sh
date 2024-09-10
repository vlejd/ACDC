#!/bin/bash
#SBATCH -J "imgnet"     # name of job in SLURM
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -o joblog/run_imagenet_iht_oneshot_cosinelr_s80_ep100.sh
#SBATCH -e joblog/run_imagenet_iht_oneshot_cosinelr_s80_ep100.sh
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
# conda env list
# conda init bash
conda activate acdc_slurm

# pip freeze


echo "Lets get this party started!"
bash run_imagenet_iht_oneshot_cosinelr_s80_ep100.sh


