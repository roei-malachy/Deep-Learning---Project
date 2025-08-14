#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Option 1: Large backbone
# python train.py -G=0 -C=r3d200 epochs=400 fold=999 save_weights=True

# # Option 2: Large backbone (less epochs)
#python train.py -G=0 -C=r3d200 epochs=250 fold=999 save_weights=True

# # Option 3: Small backbone (for testing)
python train.py -G=0 -C=r3d200 epochs=1 save_weights=False fold=0 backbone="swin3d_b"
