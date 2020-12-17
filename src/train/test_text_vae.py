import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
from src.train.eval import eval_text_vae
from src.constants import SOS, EOS, PAD_INDEX
from src.module.criterion.language_cross_entropy import LanguageCrossEntropyLoss
from src.utils.gaussian_kldiv import GaussianKLDiv
import math

def test_vae(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'text_vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]

    test_data = TabularDataset(path=os.path.join(base_path, 'test.tsv'),
                                format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=config['text_vae']['training']['batch_size'], shuffle=False, device=device)

    model = torch.load(save_path)

    criterion = LanguageCrossEntropyLoss(ignore_index=PAD_INDEX)
    kldiv = GaussianKLDiv(reduction="none")

    test_reconstruction, test_kl, test_nll, test_ppl, test_wer, forward_ppl = eval_text_vae(model, test_iter, criterion, kldiv, base_path, max_len=config['max_len'])
    print('test_reconstruction: %.4f\ttest_kl: %.4f\ttest_nll: %.4f\ttest_ppl: %.4f\ttest_wer: %.4f\tforward_ppl: %.4f' %
          (test_reconstruction, test_kl, test_nll, test_ppl, test_wer, forward_ppl))