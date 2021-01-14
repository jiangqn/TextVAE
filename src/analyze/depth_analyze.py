import os
from src.analyze.regression_analyzer import RegressionAnalyzer
import matplotlib.pyplot as plt
from hyperanalysis.visualization.lra import linear_regression_analysis

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
    # analyzer.load_model()

    latent_variable, transformed_latent_variable, target = analyzer.get_data()

    projected_latent_variable = linear_regression_analysis(latent_variable, target).cpu().numpy()
    projected_transformed_latent_variable = linear_regression_analysis(transformed_latent_variable,
                                                                       target).cpu().numpy()
    target = target.cpu().numpy()

    plt.figure(figsize=(16, 13))

    plt.subplot(2, 2, 1)

    plt.scatter(projected_latent_variable[:, 0], projected_latent_variable[:, 1], c=target, s=0.1, cmap="viridis")
    plt.colorbar()
    plt.title("latent space")

    plt.subplot(2, 2, 2)
    plt.scatter(projected_transformed_latent_variable[:, 0], projected_transformed_latent_variable[:, 1], c=target,
                s=0.1, cmap="viridis")
    plt.colorbar()
    plt.title("transformed latent space")

    plt.subplot(2, 2, 3)
    plt.hist(projected_latent_variable[:, 0], bins=45)
    plt.title("histogram")
    plt.xlabel("depth main direction")
    plt.ylabel("frequency")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(projected_transformed_latent_variable[:, 0], bins=45)
    plt.title("transformed histogram")
    plt.xlabel("depth main direction")
    plt.ylabel("frequency")
    plt.legend()

    figure_save_path = os.path.join(base_path, "depth_visualization.png")
    plt.savefig(figure_save_path)

    plt.clf()

    plt.figure(figsize=(8, 5))
    plt.hist(target, bins=45)
    plt.title("depth histogram")
    plt.xlabel("depth")
    plt.ylabel("frequency")
    distribution_figure_save_path = os.path.join(base_path, "depth_distribution.png")
    plt.savefig(distribution_figure_save_path)