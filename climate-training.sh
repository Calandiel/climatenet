#!/bin/sh

# -- job description --
#SBATCH --job-name="2node-2GPU-4CPU"

# -- resource allocation --
#SBATCH --partition=GPUQ		
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB

# -- I/O --
#SBATCH --output=climatejob_%j.out
#SBATCH --error=climatejob_%j.err

#Add the following to ignore any incidental GPUs you may have: os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Hide the GPUs from tensorflow, if any
#To requisition a live node: salloc --nodes=1 --partition=GPUQ --gres=gpu:1 --time=00:30:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,0,0

module purge
module load fosscuda/2019b
module load TensorFlow/2.1.0-Python-3.7.4
module load GDAL/3.0.2-Python-3.7.4

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Job NPROCS: ${SLURM_NPROCS}"
echo "== Job NNODES: ${SLURM_NNODES}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

srun python main.py

