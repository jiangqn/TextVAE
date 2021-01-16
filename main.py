import argparse
import yaml
import os
from src.utils.set_seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="text_vae", choices=["text_vae", "text_cnn", "lm"])
parser.add_argument("--task", type=str, default="train", choices=["preprocess", "train", "test", "vanilla_sample", "get_features",
        "length_analyze", "depth_analyze", "category_analyze", "linear_length_sample", "linear_depth_sample", "linear_categorical_sample",
        "nonlinear_length_sample", "nonlinear_depth_sample", "nonlinear_categorical_sample", "test_linear_length_sample",
        "eval_reverse_ppl", "measure_disentanglement", "length_interpolate", "compute_aggregated_posterior",
        "register_aggregated_posterior", "remove_aggregated_posterior"])
parser.add_argument("--gpu", type=int, default=0, choices=[i for i in range(8)])
parser.add_argument("--config", type=str, default="yelp_config.yaml")
parser.add_argument("--aggregated_posterior_ratio", type=float, default=None)

args = parser.parse_args()

config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
config["gpu"] = args.gpu
if args.aggregated_posterior_ratio != None:
    config["sample"]["aggregated_posterior_ratio"] = args.aggregated_posterior_ratio

set_seed(config["seed"])

if args.task == "preprocess":
    from src.train.preprocess import preprocess
    preprocess(config)
elif args.model == "text_vae":
    if args.task == "train":
        from src.train.train_text_vae import train_text_vae
        train_text_vae(config)
    elif args.task == "test":
        from src.train.test_text_vae import test_vae
        test_vae(config)
    elif args.task == "vanilla_sample":
        from src.sample.vanilla_sample import vanilla_sample
        vanilla_sample(config)
    elif args.task == "get_features":
        from src.get_features.get_features import get_features
        get_features(config)
    elif args.task == "length_analyze":
        from src.analyze.length_analyze import length_analyze
        length_analyze(config)
    elif args.task == "depth_analyze":
        from src.analyze.depth_analyze import depth_analyze
        depth_analyze(config)
    elif args.task == "category_analyze":
        from src.analyze.category_analyze import category_analyze
        category_analyze(config)
    elif args.task == "linear_length_sample":
        from src.sample.linear_length_sample import linear_length_sample
        linear_length_sample(config)
    elif args.task == "linear_depth_sample":
        from src.sample.linear_depth_sample import linear_depth_sample
        linear_depth_sample(config)
    elif args.task == "linear_categorical_sample":
        from src.sample.linear_categorical_sample import linear_categorical_sample
        linear_categorical_sample(config)
    elif args.task == "nonlinear_length_sample":
        from src.sample.nonlinear_length_sample import nonlinear_length_sample
        nonlinear_length_sample(config)
    elif args.task == "nonlinear_depth_sample":
        from src.sample.nonlinear_depth_sample import nonlinear_depth_sample
        nonlinear_depth_sample(config)
    elif args.task == "nonlinear_categorical_sample":
        from src.sample.nonlinear_categorical_sample import nonlinear_categorical_sample
        nonlinear_categorical_sample(config)
    elif args.task == "eval_reverse_ppl":
        from src.train.eval_reverse_ppl import eval_reverse_ppl
        path = os.path.join(config["base_path"], "vanilla_sample_100000.tsv")
        eval_reverse_ppl(config)
    elif args.task == "length_interpolate":
        from src.sample.length_interpolate import length_interpolate
        length_interpolate(config)
    elif args.task == "compute_aggregated_posterior":
        from src.sample.compute_aggregated_posterior import compute_aggregated_posterior
        compute_aggregated_posterior(config)
    elif args.task == "register_aggregated_posterior":
        from src.sample.register_aggregated_posterior import register_aggregated_posterior
        register_aggregated_posterior(config)
    elif args.task == "remove_aggregated_posterior":
        from src.sample.remove_aggregated_posterior import remove_aggregated_posterior
        remove_aggregated_posterior(config)
    elif args.task == "test_linear_length_sample":
        from src.sample.test_linear_length_sample import test_linear_length_sample
        test_linear_length_sample(config)
elif args.model == "lm":
    if args.task == "train":
        from src.train.train_language_model import train_language_model
        train_language_model(config)
    elif args.task == "test":
        from src.train.test_language_model import test_language_model
        test_language_model(config)
else:   # text_cnn
    if args.task == "train":
        from src.train.train_text_cnn import train_text_cnn
        train_text_cnn(config)
    elif args.task == "test":
        from src.train.test_text_cnn import test_text_cnn
        test_text_cnn(config)