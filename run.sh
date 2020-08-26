#!/bin/bash
gpu=1
echo "train"
python main.py --model vae --task train --gpu $gpu
echo "vanilla_sample"
python main.py --model vae --task vanilla_sample --gpu $gpu
echo "get_features"
python main.py --model vae --task get_features --gpu $gpu
echo "linear_separate"
python main.py --model vae --task linear_separate --gpu $gpu
echo "categorical_sample"
python main.py --model vae --task categorical_sample --gpu $gpu
echo "correlation"
python main.py --model vae --task correlation --gpu $gpu
echo "compute_projection_statistics"
python main.py --model vae --task compute_projection_statistics --gpu $gpu
echo "length_sample"
python main.py --model vae --task length_sample --gpu $gpu
echo "depth_sample"
python main.py --model vae --task depth_sample --gpu $gpu