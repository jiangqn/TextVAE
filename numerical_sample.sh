#!/bin/bash
gpu=1
echo "correlation"
python main.py --model vae --task correlation --gpu $gpu
echo "compute_projection_statistics"
python main.py --model vae --task compute_projection_statistics --gpu $gpu
echo "length_sample"
python main.py --model vae --task length_sample --gpu $gpu
echo "depth_sample"
python main.py --model vae --task depth_sample --gpu $gpu