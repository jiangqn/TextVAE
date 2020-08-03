import os
import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import logging
import pickle
from src.text_vae import TextVAE
from src.constants import PAD_INDEX, SOS, EOS

def train_vae(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    glove_source_path = config['glove_source_path']

    config = config['vae']

    logger.info('build dataset')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]
    train_data = TabularDataset(path=os.path.join(base_path, 'train.tsv'),
                                format='tsv', skip_header=True, fields=fields)
    dev_data = TabularDataset(path=os.path.join(base_path, 'dev.tsv'),
                              format='tsv', skip_header=True, fields=fields)

    logger.info('load vocab')
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab
    vocab_size = len(vocab.itos)
    logger.info('vocab_size: %d' % vocab_size)

    logger.info('build data iterator')
    device = torch.device('cuda:0')
    train_iter = Iterator(train_data, batch_size=config['batch_size'], shuffle=True, device=device)
    dev_iter = Iterator(dev_data, batch_size=config['batch_size'], shuffle=False, device=device)

    logger.info('build model')
    model = TextVAE(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        enc_dec_tying=config['enc_dec_tying'],
        dec_gen_tying=config['dec_gen_tying']
    )

    logger.info('transfer model to GPU')
    model = model.to(device)

    logger.info('set up criterion and optimizer')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    logger.info('start train')

    min_dev_loss = 1e9

    for epoch in range(config['epoches']):

        total_tokens = 0
        correct_tokens = 0
        total_loss = 0

        for i, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()

            sentence = batch.sentence
            src = sentence[:, 1:]
            trg_input = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)
            trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(src, trg_input)
            trg_output = trg_output.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, trg_output)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            optimizer.step()

            mask = (trg_output != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_loss += token_num * loss.item()
            prediction = logit.argmax(dim=-1)
            correct_tokens += (prediction.masked_select(mask) == trg_output.masked_select(mask)).long().sum().item()

            if i % config['eval_freq'] == 0:
                train_loss = total_loss / total_tokens
                train_accuracy = correct_tokens / total_tokens
                total_loss = 0
                correct_tokens = 0
                total_tokens = 0
                logger.info('[epoch %2d step %4d]\ttrain_loss: %.4f\ttrain_accuracy: %.4f' %
                            (epoch, i, train_loss, train_accuracy))
        torch.save(model, save_path)
        raise ValueError('debug')
            # loss = criterion(logit, label)
            # loss.backward()
            # optimizer.step()
            #
            # batch_size = label.size(0)
            # prediction = logit.argmax(dim=-1)
            # total_samples += batch_size
            # correct_samples += (prediction == label).long().sum().item()
            # total_loss += batch_size * loss.item()

            # if i % config['eval_freq'] == 0:
            #
            #     train_loss = total_loss / total_samples
            #     train_accuracy = correct_samples / total_samples
            #     total_samples = 0
            #     total_loss = 0
            #     correct_samples = 0
            #
            #     dev_loss, dev_accuracy = eval_text_cnn(model, dev_iter, criterion)
            #
            #     logger.info('[epoch %2d step %4d]\ttrain_loss: %.4f\ttrain_accuracy: %.4f\tdev_loss: %.4f\tdev_accuracy: %.4f' %
            #                 (epoch, i, train_loss, train_accuracy, dev_loss, dev_accuracy))
            #
            #     if dev_loss < min_dev_loss:
            #         min_dev_loss = dev_loss
            #         corr_dev_accuracy = dev_accuracy
            #         torch.save(model, save_path)