import os
import torch
import pickle
import csv
from src.utils.tsv_reader import read_field
from src.utils.multinomial_distribution import get_multinomial_distribution, sample_from_multinomial_distribution
from src.analyze.latent_variable_transform import minimal_norm_solve
from src.utils.sample_from_encoding import sample_sentences_from_latent_variable
from src.get_features.get_length import get_length
from src.utils import metric
from src.get_features.get_ppl import get_ppl_from_tsv
from src.train.eval_reverse_ppl import eval_reverse_ppl
import numpy as np

def test_linear_length_sample(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]

    vanilla_sample_save_path = os.path.join(base_path, "vanilla_sample_test.tsv")
    vanilla_sample_length = read_field(vanilla_sample_save_path, "length")
    length_distribution = get_multinomial_distribution(vanilla_sample_length)
    vanilla_sample_latent_variable_save_path = os.path.join(base_path, "vanilla_sample_test.npy")
    vanilla_sample_latent_variable = np.load(vanilla_sample_latent_variable_save_path)

    sample_num = config["sample"]["length_sample"]["sample_num"]
    sample_save_path = os.path.join(base_path, "length_sample.tsv")
    target_length = sample_from_multinomial_distribution(length_distribution, sample_num)

    print(target_length[0:100])

    vocab_path = os.path.join(base_path, "vocab.pkl")
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    model_path = os.path.join(base_path, "text_vae.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")
    latent_linear_weights_path = os.path.join(base_path, "length_latent_linear_weights.pkl")

    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device
    weights = torch.load(latent_linear_weights_path)
    weight = weights[0:-1]
    bias = weights[-1]

    weight = weight / weight.norm()

    vanilla_sample_latent_variable = torch.from_numpy(vanilla_sample_latent_variable).to(device)
    projection = vanilla_sample_latent_variable.matmul(weight)

    max_length = max(vanilla_sample_length)
    vanilla_sample_length = torch.LongTensor(vanilla_sample_length).to(device)
    projection_dict = torch.zeros(max_length + 1, dtype=torch.float, device=device)

    for i in range(1, max_length + 1):
        projection_dict[i] = projection[vanilla_sample_length == i].mean()

    projection_dict = projection_dict.cpu().numpy()
    x = [i for i in range(1, max_length + 1)]
    y = [projection_dict[i] for i in range(1, max_length + 1)]

    vanilla_sample_length = vanilla_sample_length.cpu().numpy()
    projection = projection.cpu().numpy()

    from sklearn.linear_model import LinearRegression
    from sklearn.kernel_ridge import KernelRidge

    lr = LinearRegression()
    lr.fit(vanilla_sample_length[:, np.newaxis], projection)
    z = lr.predict(np.asarray(x)[:, np.newaxis])

    import matplotlib.pyplot as plt
    plt.scatter(vanilla_sample_length, projection, c=vanilla_sample_length, s=0.1)
    plt.plot(x, y)
    plt.plot(x, z)

    plt.savefig("test.png")

    # target_projection = projection_dict[torch.LongTensor(target_length).to(device)]
    #
    # latent_variable = model.sample_latent_variable(sample_num)
    # latent_variable = latent_variable + (target_projection - latent_variable.matmul(weight)).unsqueeze(1).matmul(weight.unsqueeze(0))
    #
    # sentences = sample_sentences_from_latent_variable(model, vocab, latent_variable, config["max_len"], config["text_vae"]["training"]["batch_size"])
    # length = get_length(sentences)
    #
    # print(target_length[0:100])
    # print(length[0:100])
    #
    # sentences = ["sentence"] + sentences
    # sentences = [[sentence] for sentence in sentences]
    #
    # with open(sample_save_path, "w") as f:
    #     writer = csv.writer(f, delimiter="\t")
    #     writer.writerows(sentences)
    #
    # ppl = get_ppl_from_tsv(sample_save_path, config["language_model"]["training"]["batch_size"], model_path=language_model_path, vocab_path=vocab_path)
    #
    # print("length sample")
    # print("accuracy: %.4f" % metric.accuracy(length, target_length))
    # print("diff: %.4f" % metric.diff(length, target_length))
    # print("correlation: %.4f" % metric.correlation(length, target_length))
    # print("ppl: %.4f" % metric.mean(ppl))

    # reverse_ppl = eval_reverse_ppl(config, sample_save_path)

    # print("length sample reverse ppl: %.4f" % reverse_ppl)