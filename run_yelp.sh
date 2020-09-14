#!/bin/bash
gpu=6
echo "train text_cnn"
python main.py --config yelp_config.yaml --model text_cnn --task train --gpu $gpu
echo "test text_cnn"
python main.py --config yelp_config.yaml --model text_cnn --task test --gpu $gpu
echo "train lm"
python main.py --config yelp_config.yaml --model lm --task train --gpu $gpu
echo "test lm"
python main.py --config yelp_config.yaml --model lm --task test --gpu $gpu
echo "train"
python main.py --config yelp_config.yaml --model vae --task train --gpu $gpu
echo "vanilla_sample"
python main.py --config yelp_config.yaml --model vae --task vanilla_sample --gpu $gpu
echo "get_features"
python main.py --config yelp_config.yaml --model vae --task get_features --gpu $gpu
echo "linear_separate"
python main.py --config yelp_config.yaml --model vae --task linear_separate --gpu $gpu
echo "categorical_sample"
python main.py --config yelp_config.yaml --model vae --task categorical_sample --gpu $gpu
echo "correlation"
python main.py --config yelp_config.yaml --model vae --task correlation --gpu $gpu
echo "compute_projection_statistics"
python main.py --config yelp_config.yaml --model vae --task compute_projection_statistics --gpu $gpu
echo "length_sample"
python main.py --config yelp_config.yaml --model vae --task length_sample --gpu $gpu
echo "depth_sample"
python main.py --config yelp_config.yaml --model vae --task depth_sample --gpu $gpu
echo "measure_disentanglement"
python main.py --config yelp_config.yaml --model vae --task measure_disentanglement --gpu $gpu