CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 500 --batch-size 8 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset aicup --task segmentation --use-balanced-weights --base-size=513 --crop-size=513
