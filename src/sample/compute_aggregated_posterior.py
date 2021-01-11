import os
import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import pickle
from src.constants import PAD_INDEX, SOS, EOS
import numpy as np
from hyperanalysis.utils.linalg import cov

def compute_aggregated_posterior(config: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    save_path = os.path.join(base_path, "text_vae.pkl")
    vocab_path = os.path.join(base_path, "vocab.pkl")

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [("sentence", TEXT)]

    test_data = TabularDataset(path=os.path.join(base_path, "test.tsv"),
                               format="tsv", skip_header=True, fields=fields)
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device("cuda:0")
    test_iter = Iterator(test_data, batch_size=config["text_vae"]["training"]["batch_size"], shuffle=False,
                         device=device)

    model = torch.load(save_path)

    latent_variable_list = []

    model.eval()

    t = 0

    for batch in test_iter:
        sentence = batch.sentence
        latent_variable = model.gradient_encode(sentence)
        latent_variable_list.append(latent_variable)
        t += latent_variable.shape[0]
        print(t)

    latent_variable  = torch.cat(latent_variable_list, dim=0)

    aggregated_posterior_mean = latent_variable.mean(dim=0)
    aggregated_posterior_std = latent_variable.std(dim=0)
    aggregated_posterior_cov = cov(latent_variable)

    U, s, V = torch.svd(aggregated_posterior_cov)
    aggregated_posterior_weight = U.matmul(torch.diag(torch.sqrt(s)))
    # W = U.matmul(torch.diag(torch.sqrt(s))).matmul(U.t())

    model.register_buffer("aggregated_posterior_mean", aggregated_posterior_mean)
    model.register_buffer("aggregated_posterior_weight", aggregated_posterior_weight)
    torch.save(model, save_path)

    aggregated_posterior_mean = aggregated_posterior_mean.cpu().numpy()
    aggregated_posterior_std = aggregated_posterior_std.cpu().numpy()
    aggregated_posterior_cov = aggregated_posterior_cov.cpu().numpy()
    aggregated_posterior_weight = aggregated_posterior_weight.cpu().numpy()

    aggregated_posterior_path = os.path.join(base_path, "aggregated_posterior.npz")

    np.savez(aggregated_posterior_path, mean=aggregated_posterior_mean, std=aggregated_posterior_std,
             cov=aggregated_posterior_cov, weight=aggregated_posterior_weight)