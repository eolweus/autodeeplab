CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
 --batch-size 6 --dataset cityscapes --checkname firstTest \
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 321