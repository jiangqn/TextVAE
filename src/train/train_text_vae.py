import os
import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
from torch.utils.tensorboard import SummaryWriter
import logging
import pickle
import numpy as np
import math
from src.model.build_model import build_model
from src.constants import PAD_INDEX, SOS, EOS
from src.train.eval import eval_text_vae
from src.train.eval_reverse_ppl import eval_reverse_ppl
from src.utils.gaussian_kldiv import GaussianKLDiv
from src.utils.kl_anneal import KLAnnealer
from copy import deepcopy
from src.module.criterion.language_cross_entropy import LanguageCrossEntropyLoss
from src.utils.generate_pad import generate_pad

def train_text_vae(config: dict) -> None:

    config_copy = deepcopy(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    logger = logging.getLogger(__name__)

    base_path = config["base_path"]
    save_path = os.path.join(base_path, "text_vae.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")
    vocab_path = os.path.join(base_path, "vocab.pkl")
    embedding_path = os.path.join(base_path, "embedding.npy")
    log_base_path = os.path.join(base_path, "log")

    if os.path.isdir(log_base_path):
        os.system("rm -rf %s" % log_base_path)

    writer = SummaryWriter(log_base_path)

    max_len = config["max_len"]

    config = config["text_vae"]
    train_config = config["training"]

    logger.info("build dataset")

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [("sentence", TEXT)]
    train_data = TabularDataset(path=os.path.join(base_path, "train.tsv"),
                                format="tsv", skip_header=True, fields=fields)
    dev_data = TabularDataset(path=os.path.join(base_path, "dev.tsv"),
                              format="tsv", skip_header=True, fields=fields)

    logger.info("load vocab")
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab
    vocab_size = len(vocab.itos)
    logger.info("vocab_size: %d" % vocab_size)
    config["vocab_size"] = vocab_size

    language_model = torch.load(language_model_path)

    logger.info("build data iterator")
    device = torch.device("cuda:0")
    train_iter = Iterator(train_data, batch_size=train_config["batch_size"], shuffle=True, device=device)
    dev_iter = Iterator(dev_data, batch_size=train_config["batch_size"], shuffle=False, device=device)

    logger.info("build model")
    model = build_model(config)
    model.load_pretrained_embeddings(path=embedding_path)

    logger.info("transfer model to GPU")
    model = model.to(device)

    logger.info("set up criterion and optimizer")
    criterion = LanguageCrossEntropyLoss(ignore_index=PAD_INDEX)
    kldiv = GaussianKLDiv(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])

    logger.info("start train")

    kl_annealer = KLAnnealer(beta=train_config["beta"], anneal_step=train_config["anneal_step"])

    corr_reconstruction = 1e9
    corr_kl = 1e9
    corr_nll = 1e9
    corr_ppl = 1e9
    corr_loss = 1e9
    corr_wer = 1
    corr_forward_ppl = 1e9
    corr_epoch = 0
    corr_step = 0

    global_step = 0 # min(globel_step, config["anneal_step"]) / config["anneal_step"]

    for epoch in range(train_config["epoches"]):

        total_samples = 0
        total_tokens = 0

        total_reconstruction = 0
        total_kl = 0
        total_nll = 0
        total_loss = 0
        correct_tokens = 0

        for i, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()

            sentence = batch.sentence
            src = sentence[:, 1:]
            trg_input = sentence
            batch_size = sentence.size(0)
            pad = generate_pad(size=(batch_size, 1), device=sentence.device)
            trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit, posterior_mean, posterior_std = model(src, trg_input)

            reconstruction, seq_lens = criterion(logit, trg_output)
            kl = kldiv(posterior_mean, posterior_std)
            nll = reconstruction + kl
            loss = reconstruction + kl * kl_annealer.linear_anneal(global_step)

            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config["clip_grad_norm"])
            optimizer.step()
            global_step += 1

            total_samples += batch_size
            total_tokens += seq_lens.long().sum().item()

            total_reconstruction += reconstruction.sum().item()
            total_kl += kl.sum().item()
            total_nll += nll.sum().item()
            total_loss += batch_size * loss.item()

            mask = (trg_output != PAD_INDEX)
            prediction = logit.argmax(dim=-1)
            correct_tokens += (prediction.masked_select(mask) == trg_output.masked_select(mask)).long().sum().item()

            if i % train_config["eval_freq"] == 0:

                train_reconstruction = total_reconstruction / total_samples
                train_kl = total_kl / total_samples
                train_nll = total_nll / total_samples
                train_ppl = math.exp(total_nll / total_tokens)
                train_loss = total_loss / total_samples
                train_wer = 1 - correct_tokens / total_tokens

                total_tokens = 0
                total_samples = 0

                total_reconstruction = 0
                total_kl = 0
                total_nll = 0
                total_loss = 0
                correct_tokens = 0

                dev_reconstruction, dev_kl, dev_nll, dev_ppl, dev_wer, forward_ppl = eval_text_vae(model, dev_iter, criterion, kldiv, base_path, language_model=language_model, max_len=max_len)
                dev_loss = dev_reconstruction + dev_kl * kl_annealer.linear_anneal(global_step)
                logger.info("[epoch %2d step %4d]\ttrain_reconstruction: %.4f train_kl: %.4f train_nll: %.4f train_ppl: %.4f train_loss: %.4f train_wer: %.4f"
                    % (epoch, i, train_reconstruction, train_kl, train_nll, train_ppl, train_loss, train_wer))
                logger.info("[epoch %2d step %4d]\tdev_reconstruction: %.4f dev_kl: %.4f dev_nll: %.4f dev_ppl: %.4f dev_loss: %.4f dev_wer: %.4f forward_ppl: %.4f"
                            % (epoch, i, dev_reconstruction, dev_kl, dev_nll, dev_ppl, dev_loss, dev_wer, forward_ppl))

                writer.add_scalar("kl_weight", kl_annealer.linear_anneal(global_step), global_step)

                writer.add_scalars(
                    "reconstruction",
                    {
                        "train_reconstruction": train_reconstruction,
                        "dev_reconstruction": dev_reconstruction
                    },
                    global_step
                )

                writer.add_scalars(
                    "kl",
                    {
                        "train_kl": train_kl,
                        "dev_kl": dev_kl
                    },
                    global_step
                )

                writer.add_scalars(
                    "nll",
                    {
                        "train_nll": train_nll,
                        "dev_nll": dev_nll
                    }
                )

                writer.add_scalars(
                    "ppl",
                    {
                        "train_ppl": train_ppl,
                        "dev_ppl": dev_ppl
                    },
                    global_step
                )

                writer.add_scalars(
                    "loss",
                    {
                        "train_loss": train_loss,
                        "dev_loss": dev_loss
                    }
                )

                writer.add_scalars(
                    "wer",
                    {
                        "train_wer": train_wer,
                        "dev_wer": dev_wer
                    },
                    global_step
                )

                writer.add_scalar("forward_ppl", forward_ppl, global_step)

                if dev_nll < corr_nll:

                    corr_reconstruction = dev_reconstruction
                    corr_kl = dev_kl
                    corr_nll = dev_nll
                    corr_ppl = dev_ppl
                    corr_loss = dev_loss
                    corr_wer = dev_wer
                    corr_forward_ppl = forward_ppl
                    corr_epoch = epoch
                    corr_step = i
                    torch.save(model, save_path)

    reverse_ppl = eval_reverse_ppl(config_copy)

    logger.info("[best checkpoint] at [epoch %2d step %4d] dev_reconstruction: %.4f dev_kl: %.4f dev_nll: %.4f dev_ppl: %.4f dev_loss: %.4f dev_wer: %.4f forward_ppl: %.4f reverse_ppl: %.4f"
                % (corr_epoch, corr_step, corr_reconstruction, corr_kl, corr_nll, corr_ppl, corr_loss, corr_wer, corr_forward_ppl, reverse_ppl))
    logger.info("finish")