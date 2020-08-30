import argparse
import yaml
import os
from src.utils.set_seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vae', choices=['vae', 'text_cnn', 'lm'])
parser.add_argument('--task', type=str, default='train', choices=['preprocess', 'train', 'test', 'vanilla_sample', 'get_features', 'correlation',
        'visualize', 'pca_visualize', 'tsne_visualize', 'linear_separate', 'categorical_sample', 'compute_projection_statistics', 'sentiment_sample', 'length_sample', 'depth_sample',
        'test_vae_encoding', 'sentiment_transfer', 'eval_reverse_ppl'])
parser.add_argument('--gpu', type=int, default=0, choices=[i for i in range(8)])
parser.add_argument('--config', type=str, default='yelp_config.yaml')

args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
config['gpu'] = args.gpu

set_seed(config['seed'])

if args.task == 'preprocess':
    from src.train.preprocess import preprocess
    preprocess(config)
elif args.model == 'vae':
    if args.task == 'train':
        from src.train.train_text_vae import train_vae
        train_vae(config)
    elif args.task == 'test':
        from src.train.test_text_vae import test_vae
        test_vae(config)
    elif args.task == 'vanilla_sample':
        from src.sample.vanilla_sample import vanilla_sample
        vanilla_sample(config)
    elif args.task == 'get_features':
        from src.get_features.get_features import get_features
        get_features(config)
    elif args.task == 'correlation':
        from src.utils.correlation import correlation
        correlation(config)
    elif args.task == 'visualize':
        from src.utils.visualize import visualize
        visualize(config)
    elif args.task == 'pca_visualize':
        from src.utils.vanilla_visualize import vanilla_visualize
        vanilla_visualize(config, 'pca')
    elif args.task == 'tsne_visualize':
        from src.utils.vanilla_visualize import vanilla_visualize
        vanilla_visualize(config, 'tsne')
    elif args.task == 'linear_separate':
        from src.utils.linear_separate import linear_separate
        linear_separate(config)
    elif args.task == 'categorical_sample':
        from src.sample.categorical_sample import categorical_sample
        categorical_sample(config)
    elif args.task == 'compute_projection_statistics':
        from src.utils.compute_projection_statistics import compute_projection_statistics
        compute_projection_statistics(config)
    elif args.task == 'length_sample':
        # from src.sample.syntax_sample import syntax_sample
        # syntax_sample(config, 'length')
        from src.sample.length_sample import length_sample
        length_sample(config)
    elif args.task == 'depth_sample':
        # from src.sample.syntax_sample import syntax_sample
        # syntax_sample(config, 'depth')
        from src.sample.depth_sample import depth_sample
        depth_sample(config)
    elif args.task == 'test_vae_encoding':
        from src.train.test_vae_encoding import test_vae_encoding
        test_vae_encoding(config)
    elif args.task == 'eval_reverse_ppl':
        from src.train.eval_reverse_ppl import eval_reverse_ppl
        path = os.path.join(config['base_path'], 'vanilla_sample_100000.tsv')
        eval_reverse_ppl(config)
elif args.model == 'lm':
    if args.task == 'train':
        from src.train.train_language_model import train_language_model
        train_language_model(config)
    elif args.task == 'test':
        from src.train.test_language_model import test_language_model
        test_language_model(config)
else:   # text_cnn
    if args.task == 'train':
        from src.train.train_text_cnn import train_text_cnn
        train_text_cnn(config)
    elif args.task == 'test':
        from src.train.test_text_cnn import test_text_cnn
        test_text_cnn(config)