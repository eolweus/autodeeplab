CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
 --batch-size 2 --dataset solis --checkname solisTest \
 --alpha_epoch 20 --filter_multiplier 8 --resize 224 --crop_size 224