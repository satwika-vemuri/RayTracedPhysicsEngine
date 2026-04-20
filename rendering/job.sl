#!/bin/bash
#SBATCH --job-name=raytrace
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

module purge
module load legacy/CentOS7
module load gcc/8.5.0
module load nvidia-hpc-sdk/21.7

echo "HOST:"
hostname

echo "GPU CHECK:"
nvidia-smi

echo "CUDA DEVICES:"
echo $CUDA_VISIBLE_DEVICES

make clean
make || exit 1
./render