#!/bin/bash
#
#SBATCH -p gpu
#SBATCH --gres gpu:rtx5000:1
#SBATCH --mem=4gb

hostname
module load nvidia/10.0
printenv | grep CUDA

python demoPointProcessLeasSimulation.py "cpu"

