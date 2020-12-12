import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import logging
import pickle
from src.model.text_cnn import TextCNN
from src.train.eval import eval_text_cnn
from src.utils.tsv_max_entry_value import tsv_max_entry_value

def train_text_cnn(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    logger = logging.getLogger(__name__)

    base_path = config["base_path"]
    save_path = os.path.join(base_path, "text_cnn.pkl")
    vocab_path = os.path.join(base_path, "vocab.pkl")
    embedding_path = os.path.join(base_path, "embedding.npy")

    config = config["text_cnn"]
    model_config = config["model"]
    train_config = config["training"]

    logger.info("build dataset")

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    fields = [
        ("sentence", TEXT),
        ("label", LABEL)
    ]

    train_data = TabularDataset(path=os.path.join(base_path, "train.tsv"),
        format="tsv", skip_header=True, fields=fields)
    dev_data = TabularDataset(path=os.path.join(base_path, "dev.tsv"),
        format="tsv", skip_header=True, fields=fields)

    num_categories = tsv_max_entry_value(os.path.join(base_path, "train.tsv"), "label") + 1

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
    model = TextCNN(
        vocab_size=vocab_size,
        embed_size=model_config["embed_size"],
        kernel_sizes=model_config["kernel_sizes"],
        kernel_num=model_config["kernel_num"],
        dropout=model_config["dropout"],
        num_categories=num_categories
    )
    model.load_pretrained_embeddings(path=embedding_path)
    logger.info("transfer model to GPU")
    model = model.to(device)

    logger.info("set up criterion and optimizer")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])

    logger.info("start train")

    max_patience = train_config["max_patience"]

    min_dev_loss = 1e9
    corr_dev_accuracy = 0
    patience = 0

    for epoch in range(train_config["epoches"]):

        total_samples = 0
        correct_samples = 0
        total_loss = 0

        for i, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()

            sentence = batch.sentence
            label = batch.label

            logit = model(sentence)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()

            batch_size = label.size(0)
            prediction = logit.argmax(dim=-1)
            total_samples += batch_size
            correct_samples += (prediction == label).long().sum().item()
            total_loss += batch_size * loss.item()

            if i % train_config["eval_freq"] == 0:

                train_loss = total_loss / total_samples
                train_accuracy = correct_samples / total_samples
                total_samples = 0
                total_loss = 0
                correct_samples = 0

                dev_loss, dev_accuracy = eval_text_cnn(model, dev_iter, criterion)

                logger.info("[epoch %2d step %4d]\ttrain_loss: %.4f\ttrain_accuracy: %.4f\tdev_loss: %.4f\tdev_accuracy: %.4f" %
                            (epoch, i, train_loss, train_accuracy, dev_loss, dev_accuracy))

                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    corr_dev_accuracy = dev_accuracy
                    torch.save(model, save_path)
                    patience = 0
                else:
                    patience += 1

                if patience == max_patience:
                    break

        if patience == max_patience:
            break


    logger.info("dev_loss: %.4f\tdev_accuracy: %.4f" % (min_dev_loss, corr_dev_accuracy))
    logger.info("finish")