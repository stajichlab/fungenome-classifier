#!/usr/bin/bash -l
#SBATCH -c 64 --mem 256gb -p exfab --out logs/make_features.log

module load cuda/12.8 cudnn
CPU=1
if [ ! -z $SLURM_CPUS_ON_NODE ]; then
	CPU=$SLURM_CPUS_ON_NODE
fi

~/.pixi/bin/pixi run make features N_JOBS=$CPU
