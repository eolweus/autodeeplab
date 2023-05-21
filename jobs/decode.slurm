#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="P100|A100"
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15000
#SBATCH --job-name="decode-model"
#SBATCH --output=/cluster/home/erlingfo/autodeeplab/out/decode.out
#SBATCH --mail-user=erlingfo@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


# Now, adjust your batch size depending on the GPU memory size.
# This is just an example; you'd adjust this condition and action to fit your needs.

module purge
source ~/.bashrc
# module load Anaconda3/2022.05
module load CUDA/11.3.1
conda activate autodl

checkpoint=$1

export PYTHONBUFFERED=1
python /cluster/home/erlingfo/autodeeplab/decode_autodeeplab.py --resume "$checkpoint"
uname -a 

