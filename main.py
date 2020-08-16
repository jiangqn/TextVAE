import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vae', choices=['vae', 'text_cnn', 'lm'])
parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'sample',
    'predict', 'correlation', 'visualize', 'pca_visualize', 'linear_separate'])
parser.add_argument('--gpu', type=int, default=0, choices=[i for i in range(8)])
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
config['gpu'] = args.gpu

if args.model == 'vae':
    if args.task == 'train':
        from src.train.train_text_vae import train_vae
        train_vae(config)
    elif args.task == 'test':
        from src.train.test_text_vae import test_vae
        test_vae(config)
    elif args.task == 'sample':
        from src.sample_from_vae import sample_from_vae
        sample_from_vae(config)
    elif args.task == 'predict':
        from src.predict import predict
        predict(config)
    elif args.task == 'correlation':
        from src.correlation import correlation
        correlation(config)
    elif args.task == 'visualize':
        from src.visualize import visualize
        visualize(config)
    elif args.task == 'linear_separate':
        from src.linear_separate import linear_separate
        linear_separate(config)
    else:
        from src.pca_visualize import pca_visualize
        pca_visualize(config)
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