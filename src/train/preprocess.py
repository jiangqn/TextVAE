from torchtext import data
from torchtext.data import TabularDataset
import os
import logging
import pickle
import numpy as np
from src.utils.load_glove import load_glove
from src.constants import UNK, PAD, SOS, EOS

def preprocess(config: dict) -> None:

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)

    base_path = config['base_path']
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    embedding_path = os.path.join(base_path, 'embedding.npy')
    glove_source_path = config['glove_source_path']

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    fields = [('sentence', TEXT)]

    train_data = TabularDataset(path=os.path.join(base_path, 'train.tsv'),
                                format='tsv', skip_header=True, fields=fields)

    logger.info('build vocabulary')
    TEXT.build_vocab(train_data, specials=[UNK, PAD, SOS, EOS])
    vocab = TEXT.vocab
    vocab_size = len(vocab.itos)
    logger.info('vocab_size: %d' % vocab_size)
    logger.info('save vocabulary')
    with open(vocab_path, 'wb') as handle:
        pickle.dump(vocab, handle)
    logger.info('load pretrained embedding')
    embedding = load_glove(glove_source_path, vocab_size, vocab.stoi)
    logger.info('save pretrained embedding')
    np.save(embedding_path, embedding)
    logger.info('finish')