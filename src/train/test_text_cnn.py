import torch
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
from src.train.eval import eval_text_cnn

def test_text_cnn(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    save_path = os.path.join(base_path, "text_cnn.pkl")
    vocab_path = os.path.join(base_path, "vocab.pkl")

    config = config["text_cnn"]

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    fields = [
        ("sentence", TEXT),
        ("label", LABEL)
    ]

    test_data = TabularDataset(path=os.path.join(base_path, "test.tsv"),
                                format="tsv", skip_header=True, fields=fields)
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device("cuda:0")
    test_iter = Iterator(test_data, batch_size=config["training"]["batch_size"], shuffle=False, device=device)

    model = torch.load(save_path)

    test_accuracy = eval_text_cnn(model, test_iter)
    print("test_accuracy: %.4f" % test_accuracy)