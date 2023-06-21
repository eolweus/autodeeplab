#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --job-name="best_5k_retrain"
#SBATCH --output="/cluster/home/erlingfo/autodeeplab/out/best_5k_retrain.out"
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

GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

# Now, adjust your batch size depending on the GPU memory size.
# This is just an example; you'd adjust this condition and action to fit your needs.
if [[ $GPU_MEMORY -gt 80000 ]]; then
    BATCH_SIZE=280
else
    BATCH_SIZE=140
fi

# Clear and create the gpu_usage.log file
truncate -s 0 /cluster/home/erlingfo/autodeeplab/out/best_5k_retrain_gpu_usage.log
truncate -s 0 /cluster/home/erlingfo/autodeeplab/out/best_5k_retrain_cpu_usage.log

# Monitor GPU usage and log to a file
while true; do
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> /cluster/home/erlingfo/autodeeplab/out/best_5k_retrain_gpu_usage.log
    sleep 0.1
done &
GPU_MONITOR_PID=$!

# Monitor CPU usage and log to a file
while true; do
    cpu_memory=$(free -m | awk 'NR==2{printf "Memory: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }')
    echo "$cpu_memory" >> /cluster/home/erlingfo/autodeeplab/out/best_5k_retrain_cpu_usage.log
    sleep 1
done &
CPU_MONITOR_PID=$!

module purge
source ~/.bashrc
# module load Anaconda3/2022.05
module load CUDA/11.3.1
conda activate autodl

export PYTHONBUFFERED=1
CUDA_VISIBLE_DEVICES=0 python ../train.py \
 --batch_size $BATCH_SIZE --dataset solis --checkname "best_5k_retrain" \
 --resize 224 --crop_size 224 --num_bands 12 --epochs 100 --workers 10 \
 --net_arch /cluster/home/erlingfo/autodeeplab/run/solis/5k/network_path.npy \
 --cell_arch /cluster/home/erlingfo/autodeeplab/run/solis/5k/genotype.npy \
#  --resume /cluster/home/erlingfo/autodeeplab/run/solis/best_5k_retrain/experiment_0/checkpoint_6.pth.tar 

# Stop the background GPU monitor process
kill $GPU_MONITOR_PID
kill $CPU_MONITOR_PID

uname -a 

