#!/bin/bash
gpu=1
echo "linear_separate"
python main.py --model vae --task linear_separate --gpu $gpu
echo "categorical_sample"
python main.py --model vae --task categorical_sample --gpu $gpu