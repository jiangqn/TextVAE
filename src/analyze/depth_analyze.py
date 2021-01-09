import os
from src.analyze.regression_analyzer import RegressionAnalyzer

def depth_analyze(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    latent_size = config["text_vae"]["latent_size"]

    analyzer = RegressionAnalyzer(
        base_path=base_path,
        target_name="depth",
        hidden_size=latent_size,
        n_blocks=3
    )

    analyzer.fit(num_epoches=20)