import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vae', choices=['vae', 'text_cnn'])
parser.add_argument('--task', type=str, default='train', choices=['train', 'dev', 'test', 'sample'])
parser.add_argument('--gpu', type=int, default=0, choices=[i for i in range(8)])
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
config['gpu'] = args.gpu

if args.model == 'vae':
    if args.task == 'train':
        from src.train_vae import train_vae
        train_vae(config)
    elif args.task == 'test':
        pass
else:   # text_cnn
    if args.task == 'train':
        from src.train_text_cnn import train_text_cnn
        train_text_cnn(config)
    elif args.task == 'test':
        from src.test_text_cnn import test_text_cnn
        test_text_cnn(config)