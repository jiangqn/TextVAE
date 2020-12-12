import os
import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import logging
import pickle
from src.model.language_model import LanguageModel
from src.constants import PAD_INDEX, SOS, EOS
from src.train.eval import eval_language_model
import math

def train_language_model(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    logger = logging.getLogger(__name__)

    base_path = config["base_path"]
    save_path = os.path.join(base_path, "language_model.pkl")
    vocab_path = os.path.join(base_path, "vocab.pkl")
    embedding_path = os.path.join(base_path, "embedding.npy")

    config = config["language_model"]
    model_config = config["model"]
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

    logger.info("build data iterator")
    device = torch.device("cuda:0")
    train_iter = Iterator(train_data, batch_size=train_config["batch_size"], shuffle=True, device=device)
    dev_iter = Iterator(dev_data, batch_size=train_config["batch_size"], shuffle=False, device=device)

    logger.info("build model")
    model = LanguageModel(
        vocab_size=vocab_size,
        embed_size=model_config["embed_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        weight_tying=model_config["weight_tying"]
    )
    model.load_pretrained_embeddings(path=embedding_path)

    logger.info("transfer model to GPU")
    model = model.to(device)

    logger.info("set up criterion and optimizer")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])

    logger.info("start train")

    min_dev_loss = 1e9

    for epoch in range(train_config["epoches"]):

        total_tokens = 0
        total_loss = 0

        for i, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()

            sentence = batch.sentence
            input_sentence = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(input_sentence)
            output_sentence = output_sentence.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, output_sentence)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config["clip_grad_norm"])
            optimizer.step()

            mask = (output_sentence != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_loss += token_num * loss.item()

            if i % train_config["eval_freq"] == 0:
                train_loss = total_loss / total_tokens
                total_loss = 0
                total_tokens = 0
                dev_loss = eval_language_model(model, dev_iter, criterion)
                logger.info("[epoch %2d step %4d]\ttrain_loss: %.4f train_ppl: %.4f dev_loss: %.4f dev_ppl: %.4f" %
                            (epoch, i, train_loss, math.exp(train_loss), dev_loss, math.exp(dev_loss)))

                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    torch.save(model, save_path)

    logger.info("dev_loss: %.4f\tdev_ppl: %.4f" % (min_dev_loss, math.exp(min_dev_loss)))
    logger.info("finish")