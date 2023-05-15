#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15000
#SBATCH --job-name="2k"
#SBATCH --output=~/autodeeplab/out/2k.out
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

# Clear and create the gpu_usage.log file
truncate -s 0 gpu_usage_2k.log

# Monitor GPU usage and log to a file
while true; do
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu_usage_2k.log
    sleep 10
done &
GPU_MONITOR_PID=$!

module purge
source ~/.bashrc
# module load Anaconda3/2022.05
module load CUDA/11.3.1
conda activate autodl
export PYTHONBUFFERED=1


resume_flag=""
while getopts ":r:" opt; do
  case ${opt} in
    r )
      resume_flag="--resume ${OPTARG}"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
  --batch-size 10 \
  --dataset solis \
  --checkname no_bb_2040 \
  --alpha_epoch 20 \
  --filter_multiplier 8 \
  --resize 224 \
  --crop_size 224 \
  --num_bands 12 \
  --epochs 40 \
  --num_images 2040 \
  --loss-type focal \
  --use_amp \
  $resume_flag

# Stop the background GPU monitor process
kill $GPU_MONITOR_PID

uname -a 
