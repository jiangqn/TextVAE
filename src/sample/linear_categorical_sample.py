import os
import torch
import numpy as np
import pickle
import csv
import time
from src.sample.sample_by_latent_variable import sample_by_latent_variable
from src.get_features.get_category import get_categorical_features_from_tsv
from src.get_features.get_ppl import get_ppl_from_tsv
from src.utils import metric
from src.train.eval_reverse_ppl import eval_reverse_ppl
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution

def linear_categorical_sample(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]

    vanilla_sample_save_path = os.path.join(base_path, "vanilla_sample_test.tsv")
    vanilla_sample_category = read_field(vanilla_sample_save_path, "label")
    category_distribution = get_multinomial_distribution(vanilla_sample_category)

    sample_num = config["sample"]["categorical_sample"]["sample_num"]
    sample_save_path = os.path.join(base_path, "categorical_sample.tsv")
    target_category = sample_from_multinomial_distribution(category_distribution, sample_num)

    vocab_path = os.path.join(base_path, "vocab.pkl")
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    model_path = os.path.join(base_path, "text_vae.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")
    text_cnn_path = os.path.join(base_path, "text_cnn.pkl")
    analyzer_path = os.path.join(base_path, "category_analyzer.pkl")

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    with open(analyzer_path, "rb") as f:
        analyzer = pickle.load(f)

    start = time.time()
    confidence = config["sample"]["categorical_sample"].get("confidence", None)
    if confidence == None:
        latent_variable = analyzer.rejection_sample(text_vae=model, target=torch.tensor(target_category).to(device))
    else:
        latent_variable = analyzer.rejection_sample_with_confidence(text_vae=model, target=torch.tensor(target_category).to(device), confidence_threshold=confidence)
    end = time.time()
    print("latent variable sample time: %.4f s" % (end - start))

    start = time.time()
    sentences = sample_by_latent_variable(model, vocab, latent_variable, config["max_len"],
                                          config["text_vae"]["training"]["batch_size"])
    end = time.time()
    print("sentence generation time: %.4f s" % (end - start))

    sentences = ["sentence"] + sentences
    sentences = [[sentence] for sentence in sentences]

    with open(sample_save_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(sentences)

    category = get_categorical_features_from_tsv(sample_save_path, config["text_cnn"]["training"]["batch_size"], vocab=vocab, model_path=text_cnn_path, output_category=True)
    ppl = get_ppl_from_tsv(sample_save_path, config["language_model"]["training"]["batch_size"],
                           model_path=language_model_path, vocab_path=vocab_path)

    print("categorical sample")
    print("accuracy: %.4f" % metric.accuracy(category, target_category))
    print("ppl: %.4f" % metric.mean(ppl))
    reverse_ppl = eval_reverse_ppl(config, sample_save_path)
    print("reverse ppl: %.4f" % reverse_ppl)