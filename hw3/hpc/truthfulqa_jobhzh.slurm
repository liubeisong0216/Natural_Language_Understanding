#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --account=ds_ga_1012-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=truthfulqa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh1461@nyu.edu
#SBATCH --output="truthfulqa.out"
#SBATCH --error=truthfulqa.err

module purge
cd /scratch/zh1461/hw3/hzh_hw3
OVERLAY_FILE=/scratch/zh1461/overlay-25GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/bl1986/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

singularity exec --nv \
		--bind /scratch/zh1461/ \
	    --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	    /bin/bash -c "
		source /ext3/miniconda3/etc/profile.d/conda.sh
		conda activate
		cd /scratch/zh1461/NLU/
		bash hpc/run_truthfulqa_evaluation.sh
	"