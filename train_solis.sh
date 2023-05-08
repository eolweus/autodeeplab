CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
 --batch-size 4 --dataset solis --checkname no_bb_5008 \
 --alpha_epoch 5 --filter_multiplier 8 --resize 224 --crop_size 224 \
 --num_bands 12 --epochs 30 --num_images 5008 --loss-type focal