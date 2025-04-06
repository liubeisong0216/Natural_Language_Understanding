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
#SBATCH --mail-user=bl1986@nyu.edu
#SBATCH --output="hpc_test.out"
#SBATCH --error=hpc_test.err

module purge
cd /home/bl1986/Natural_Language_Understanding/hw3
OVERLAY_FILE=/scratch/bl1986/test_singularity/overlay-25GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/bl1986/test_singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

singularity exec --nv \
		--bind /home/bl1986/Natural_Language_Understanding/hw3 \
	    --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	    /bin/bash -c "
		source /ext3/miniconda3/etc/profile.d/conda.sh
		conda activate
		bash /home/bl1986/Natural_Language_Understanding/hw3/hpc/run_test.sh
	"