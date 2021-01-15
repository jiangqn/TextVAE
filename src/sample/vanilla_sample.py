import os
import torch
import numpy as np
import pickle
import csv
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts
from src.get_features.get_ppl import get_ppl_from_tsv
from src.utils import metric
from src.train.eval_reverse_ppl import eval_reverse_ppl
from src.sample.sample_by_num import sample_by_num

def vanilla_sample(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]

    vanilla_sample_config = config["sample"]["vanilla_sample"]

    sample_dict = {
        "num": {}, "sample_path": {}, "latent_variable_path": {}
    }

    for division in ["train", "dev", "test"]:
        sample_dict["num"][division] = vanilla_sample_config["%s_sample_num" % division]
        sample_dict["sample_path"][division] = os.path.join(base_path, "vanilla_sample_%s.tsv" % division)
        sample_dict["latent_variable_path"][division] = os.path.join(base_path, "vanilla_sample_%s.npy" % division)

    model_path = os.path.join(base_path, "text_vae.pkl")
    vocab_path = os.path.join(base_path, "vocab.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")

    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    model = torch.load(model_path)

    batch_size = config["text_vae"]["training"]["batch_size"]

    for division in ["train", "dev", "test"]:

        sample_num = sample_dict["num"][division]
        sentences, latent_variables = sample_by_num(model, vocab, sample_num, max_len=config["max_len"], batch_size=batch_size)

        sentences = ["sentence"] + sentences
        sentences = [[sentence] for sentence in sentences]

        with open(sample_dict["sample_path"][division], "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(sentences)

        np.save(sample_dict["latent_variable_path"][division], latent_variables)

        if division == "test":

            ppl = get_ppl_from_tsv(sample_dict["sample_path"][division], config["language_model"]["training"]["batch_size"], model_path=language_model_path, vocab_path=vocab_path)
            print("vanilla sample ppl: %.4f" % metric.mean(ppl))

            reverse_ppl = eval_reverse_ppl(config, sample_dict["sample_path"][division])
            print("vanilla sample reverse ppl: %.4f" % reverse_ppl)