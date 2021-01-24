import os
import torch
import pickle
import csv
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution
from src.analyze.latent_variable_transform import minimal_norm_solve
from src.sample.sample_by_latent_variable import sample_by_latent_variable
from src.get_features.get_depth import get_depth
from src.utils import metric
from src.get_features.get_ppl import get_ppl_from_tsv
from src.train.eval_reverse_ppl import eval_reverse_ppl


def linear_depth_sample(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]

    vanilla_sample_save_path = os.path.join(base_path, "vanilla_sample_test.tsv")
    vanilla_sample_depth = read_field(vanilla_sample_save_path, "depth")
    depth_distribution = get_multinomial_distribution(vanilla_sample_depth)

    sample_num = config["sample"]["depth_sample"]["sample_num"]
    sample_save_path = os.path.join(base_path, "depth_sample.tsv")
    target_depth = sample_from_multinomial_distribution(depth_distribution, sample_num)

    vocab_path = os.path.join(base_path, "vocab.pkl")
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    model_path = os.path.join(base_path, "text_vae.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")
    analyzer_path = os.path.join(base_path, "depth_analyzer.pkl")

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    with open(analyzer_path, "rb") as f:
        analyzer = pickle.load(f)

    latent_variable = model.sample_latent_variable(sample_num)
    latent_variable = analyzer.latent_variable_transform(latent_variable, torch.LongTensor(target_depth).to(device))

    sentences = sample_by_latent_variable(model, vocab, latent_variable, config["max_len"], config["text_vae"]["training"]["batch_size"])
    depth = get_depth(sentences)

    sentences = ["sentence"] + sentences
    sentences = [[sentence] for sentence in sentences]

    with open(sample_save_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(sentences)

    ppl = get_ppl_from_tsv(sample_save_path, config["language_model"]["training"]["batch_size"], model_path=language_model_path, vocab_path=vocab_path)

    print("depth sample")
    print("accuracy: %.4f" % metric.accuracy(depth, target_depth))
    print("MAE: %.4f" % metric.MAE(depth, target_depth))
    print("ppl: %.4f" % metric.mean(ppl))
    reverse_ppl = eval_reverse_ppl(config, sample_save_path)
    print("reverse ppl: %.4f" % reverse_ppl)