CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
 --batch-size 6 --dataset solis --checkname no_bb_5010 \
 --alpha_epoch 20 --filter_multiplier 8 --resize 224 --crop_size 224 \
 --num_bands 12 --epochs 40 --num_images 5010