#!/bin/bash
gpu=4
export CUDA_VISIBLE_DEVICES=$gpu
echo "register_aggregated_posterior"
python main.py --model text_vae --task register_aggregated_posterior --config amazon_config.yaml --gpu $gpu
echo "vanilla_sample"
python main.py --model text_vae --task vanilla_sample --config amazon_config.yaml --gpu $gpu
echo "get_features"
python main.py --model text_vae --task get_features --config amazon_config.yaml --gpu $gpu
echo "category_analyze"
python main.py --model text_vae --task category_analyze --config amazon_config.yaml --gpu $gpu
echo "length_analyze"
python main.py --model text_vae --task length_analyze --config amazon_config.yaml --gpu $gpu
echo "depth_analyze"
python main.py --model text_vae --task depth_analyze --config amazon_config.yaml --gpu $gpu
echo "linear_categorical_sample"
python main.py --model text_vae --task linear_categorical_sample --config amazon_config.yaml --gpu $gpu
echo "linear_length_sample"
python main.py --model text_vae --task linear_length_sample --config amazon_config.yaml --gpu $gpu
echo "linear_depth_sample"
python main.py --model text_vae --task linear_depth_sample --config amazon_config.yaml --gpu $gpu