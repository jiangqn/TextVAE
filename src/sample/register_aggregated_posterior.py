import os
import torch
import numpy as np

def register_aggregated_posterior(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    base_path = config["base_path"]

    aggregated_posterior_ratio = config["sample"]["aggregated_posterior_ratio"]

    model_path = os.path.join(base_path, "text_vae.pkl")
    model = torch.load(model_path)
    device = model.encoder.embedding.weight.device

    aggregated_posterior_path = os.path.join(base_path, "aggregated_posterior.npz")
    aggregated_posterior = np.load(aggregated_posterior_path)

    posterior_mean = torch.from_numpy(aggregated_posterior["mean"]).to(device)
    posterior_std = torch.from_numpy(aggregated_posterior["std"]).to(device)
    aggregated_posterior_ratio = torch.tensor(aggregated_posterior_ratio).float().to(device)

    model.register_buffer("posterior_mean", posterior_mean)
    model.register_buffer("posterior_std", posterior_std)
    model.register_buffer("aggregated_posterior_ratio", aggregated_posterior_ratio)

    torch.save(model, model_path)