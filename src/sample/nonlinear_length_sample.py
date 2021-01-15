import os
import torch
import pickle
import csv
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution
from src.analyze.latent_variable_transform import minimal_norm_solve, transform, inverse_transform
from src.sample.sample_from_encoding import sample_sentences_from_latent_variable
from src.get_features.get_length import get_length
from src.utils import metric
from src.get_features.get_ppl import get_ppl_from_tsv


def nonlinear_length_sample(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]

    vanilla_sample_save_path = os.path.join(base_path, "vanilla_sample_test.tsv")
    vanilla_sample_length = read_field(vanilla_sample_save_path, "length")
    length_distribution = get_multinomial_distribution(vanilla_sample_length)

    sample_num = config["sample"]["length_sample"]["sample_num"]
    sample_save_path = os.path.join(base_path, "length_sample.tsv")
    target_length = sample_from_multinomial_distribution(length_distribution, sample_num)

    print(target_length[0:100])

    vocab_path = os.path.join(base_path, "vocab.pkl")
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    model_path = os.path.join(base_path, "text_vae.pkl")
    iresnet_path = os.path.join(base_path, "length_iresnet.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")
    transformed_latent_linear_weights_path = os.path.join(base_path, "length_transformed_latent_linear_weights.pkl")

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    iresnet = torch.load(iresnet_path)
    weights = torch.load(transformed_latent_linear_weights_path)
    weight = weights[0:-1]
    bias = weights[-1]

    latent_variable = model.sample_latent_variable(sample_num)
    transformed_latent_variable = transform(iresnet, latent_variable)
    inverse_transformed_latent_variable = inverse_transform(iresnet, transformed_latent_variable)

    difference = torch.norm(latent_variable - inverse_transformed_latent_variable, dim=-1).mean().item()
    print(difference)

    raise ValueError("debug")
    transformed_latent_variable = minimal_norm_solve(transformed_latent_variable, torch.FloatTensor(target_length).to(device), weight, bias)
    latent_variable = inverse_transform(iresnet, transformed_latent_variable)

    prediction = transformed_latent_variable.matmul(weight) + bias
    print(prediction.tolist()[0:100])

    sentences = sample_sentences_from_latent_variable(model, vocab, latent_variable, config["max_len"], config["text_vae"]["training"]["batch_size"])
    length = get_length(sentences)

    print(target_length[0:100])
    print(length[0:100])

    sentences = ["sentence"] + sentences
    sentences = [[sentence] for sentence in sentences]

    with open(sample_save_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(sentences)

    ppl = get_ppl_from_tsv(sample_save_path, config["language_model"]["training"]["batch_size"], model_path=language_model_path, vocab_path=vocab_path)

    print("length sample")
    print("accuracy: %.4f" % metric.accuracy(length, target_length))
    print("diff: %.4f" % metric.diff(length, target_length))
    print("correlation: %.4f" % metric.correlation(length, target_length))
    print("ppl: %.4f" % metric.mean(ppl))

    # reverse_ppl = eval_reverse_ppl(config, sample_save_path)

    # print("length sample reverse ppl: %.4f" % reverse_ppl)