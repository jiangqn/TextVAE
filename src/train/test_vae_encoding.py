import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
from nltk.translate.bleu_score import corpus_bleu
from src.constants import SOS, EOS
from src.utils.sentence_encoding import sentence_encoding_from_iterator
from src.sample.sample_from_encoding import sample_sentences_from_latent_variable
from src.utils.tsv_reader import read_field

def test_vae_encoding(config: dict) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    test_path = os.path.join(base_path, 'test.tsv')
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]

    test_data = TabularDataset(path=test_path,
                               format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=config['vae']['batch_size'], shuffle=False, device=device)
    model = torch.load(save_path)

    test_encoding = sentence_encoding_from_iterator(model, test_iter, encoding_type='gradient')
    reconstructed_sentences = sample_sentences_from_latent_variable(model, vocab, test_encoding, config['max_len'], config['vae']['batch_size'])

    original_sentences = read_field(test_path, 'sentence')

    hypothesis = [sentence.split() for sentence in reconstructed_sentences]
    references = [[sentence.split()] for sentence in original_sentences]

    bleu = corpus_bleu(references, hypothesis)
    print('bleu: %.4f' % bleu)