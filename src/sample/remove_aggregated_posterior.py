import os
import torch

def remove_aggregated_posterior(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    base_path = config["base_path"]

    model_path = os.path.join(base_path, "text_vae.pkl")
    model = torch.load(model_path)

    del model.posterior_mean
    del model.posterior_std
    del model.aggregated_posterior_ratio

    torch.save(model, model_path)