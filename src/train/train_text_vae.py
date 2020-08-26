import os
import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import logging
import pickle
import numpy as np
from src.model.text_vae import TextVAE
from src.constants import PAD_INDEX, SOS, EOS
from src.train.eval import eval_text_vae
from src.utils.gaussian_kldiv import GaussianKLDiv

def train_vae(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    language_model_path = os.path.join(base_path, 'language_model.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    embedding_path = os.path.join(base_path, 'embedding.npy')

    max_len = config['max_len']

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

    language_model = torch.load(language_model_path)

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
        word_dropout=config['word_dropout'],
        enc_dec_tying=config['enc_dec_tying'],
        dec_gen_tying=config['dec_gen_tying']
    )
    model.load_pretrained_embeddings(path=embedding_path)

    logger.info('transfer model to GPU')
    model = model.to(device)

    logger.info('set up criterion and optimizer')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    kldiv = GaussianKLDiv()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    logger.info('start train')

    # min_total_ppl = 1e9
    min_dev_loss = 1e9
    corr_ce_loss = 1e9
    corr_kl_loss = 1e9
    corr_wer = 1
    corr_sample_ppl = 1e9
    corr_epoch = 0
    corr_step = 0

    global_step = 0 # min(globel_step, config['anneal_step']) / config['anneal_step']

    for epoch in range(config['epoches']):

        total_tokens = 0
        total_samples = 0

        correct_tokens = 0
        total_ce_loss = 0
        total_kl_loss = 0

        for i, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()

            sentence = batch.sentence
            src = sentence[:, 1:]
            trg_input = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit, mean, std = model(src, trg_input)
            trg_output = trg_output.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)

            ce_loss = criterion(logit, trg_output)
            kl_loss = kldiv(mean, std)
            coefficient = config['beta'] * min(global_step, config['anneal_step']) / config['anneal_step']
            loss = ce_loss + kl_loss * coefficient
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            optimizer.step()
            global_step += 1

            mask = (trg_output != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_samples += batch_size

            total_ce_loss += token_num * ce_loss.item()
            total_kl_loss += batch_size * kl_loss.item()
            prediction = logit.argmax(dim=-1)
            correct_tokens += (prediction.masked_select(mask) == trg_output.masked_select(mask)).long().sum().item()

            if i % config['eval_freq'] == 0:
                train_wer = 1 - correct_tokens / total_tokens
                train_ce_loss = total_ce_loss / total_tokens
                train_kl_loss = total_kl_loss / total_samples
                correct_tokens = 0
                total_ce_loss = 0
                total_kl_loss = 0
                total_tokens = 0
                total_samples = 0

                dev_ce_loss, dev_kl_loss, dev_wer, sample_ppl = eval_text_vae(model, dev_iter, base_path, language_model=language_model, max_len=max_len)
                logger.info('[epoch %2d step %4d]\ttrain_ce_loss: %.4f train_kl_loss: %.4f train_ppl: %.4f train_wer: %.4f dev_ce_loss: %.4f dev_kl_loss: %.4f dev_ppl: %.4f dev_wer: %.4f sample_ppl: %.4f'
                            % (epoch, i, train_ce_loss, train_kl_loss, 2 ** train_ce_loss, train_wer, dev_ce_loss, dev_kl_loss, 2 ** dev_ce_loss, dev_wer, sample_ppl))

                dev_loss = dev_ce_loss + dev_kl_loss * config['lambd']
                # dev_ppl = 2 ** dev_ce_loss
                # total_ppl = dev_ppl + sample_ppl
                # if total_ppl < min_total_ppl:
                #     min_total_ppl = total_ppl
                if global_step > 1000 and dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    corr_ce_loss = dev_ce_loss
                    corr_kl_loss = dev_kl_loss
                    corr_wer = dev_wer
                    corr_sample_ppl = sample_ppl
                    corr_epoch = epoch
                    corr_step = i
                    torch.save(model, save_path)

    logger.info('[best checkpoint] at [epoch %2d step %4d] dev_ce_loss: %.4f dev_kl_loss: %.4f dev_ppl: %.4f dev_wer: %.4f sample_ppl: %.4f'
                % (corr_epoch, corr_step, corr_ce_loss, corr_kl_loss, 2 ** corr_ce_loss, corr_wer, corr_sample_ppl))
    logger.info('finish')