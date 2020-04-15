#!/bin/sh
#SBATCH --job-name="2GPU-4CPU"	#The name that shows up in the queue
#SBATCH --partition=GPUQ		
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB

#Add the following to ignore any incidental GPUs you may have: os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Hide the GPUs from tensorflow, if any
#To requisition a live node: salloc --nodes=1 --partition=GPUQ --gres=gpu:1 --time=00:30:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,0,0

module purge
module load fosscuda/2019b
module load TensorFlow/2.1.0-Python-3.7.4
module load GDAL/3.0.2-Python-3.7.4

srun python main.py
 
