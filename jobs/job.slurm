#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100m80:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18000
#SBATCH --job-name="10k"
#SBATCH --output="/cluster/home/erlingfo/autodeeplab/out/test-srun.out"
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
    BATCH_SIZE=24
else
    BATCH_SIZE=12
fi

# Clear and create the gpu_usage.log file
truncate -s 0 /cluster/home/erlingfo/autodeeplab/out/10k_gpu_usage.log
truncate -s 0 /cluster/home/erlingfo/autodeeplab/out/10k_cpu_usage.log

# Monitor GPU usage and log to a file
while true; do
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> /cluster/home/erlingfo/autodeeplab/out/gpu_usage.log
    sleep 10
done &
GPU_MONITOR_PID=$!

# Monitor CPU usage and log to a file
while true; do
    top -b -n 1 | grep 'Cpu(s)' >> /cluster/home/erlingfo/autodeeplab/out/10k_cpu_usage.log
    sleep 10
done &
CPU_MONITOR_PID=$!


module purge
source ~/.bashrc
# module load Anaconda3/2022.05
module load CUDA/11.3.1
conda activate autodl

export PYTHONBUFFERED=1
CUDA_VISIBLE_DEVICES=0 python /cluster/home/erlingfo/autodeeplab/train_autodeeplab.py \
 --batch-size $BATCH_SIZE --dataset solis --checkname "no_bb_10010" \
 --alpha_epoch 20 --filter_multiplier 8 --resize 224 --crop_size 224 \
 --num_bands 12 --epochs 40 --num_images 10010 --loss-type ce \
 --resume /cluster/home/erlingfo/autodeeplab/run/solis/no_bb_10010/experiment_1/checkpoint.pth.tar

# Stop the background GPU monitor process
kill $GPU_MONITOR_PID
kill $CPU_MONITOR_PID

uname -a 

