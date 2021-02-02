import os
from src.analyze.numerical_attribute_analyzer import NumericalAttributeAnalyzer
import matplotlib.pyplot as plt
from hyperanalysis.visualization.lra import linear_regression_analysis
import pickle
import numpy as np

def length_analyze(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    latent_size = config["text_vae"]["latent_size"]

    analyzer = NumericalAttributeAnalyzer(
        base_path=base_path,
        target="length",
        latent_size=latent_size
    )

    fontsize = 26

    analyzer.fit()
    analyzer_path = os.path.join(base_path, "length_analyzer.pkl")
    with open(analyzer_path, "wb") as f:
        pickle.dump(analyzer, f)

    latent_variable, target = analyzer.get_data(division="test")

    projected_latent_variable = linear_regression_analysis(latent_variable, target)
    projected_latent_variable = projected_latent_variable.cpu().numpy()
    target = target.cpu().numpy()

    min_value = int(target.min())
    max_value = int(target.max())

    plt.figure(figsize=(10, 7))

    plt.scatter(projected_latent_variable[:, 0], projected_latent_variable[:, 1], c=target, s=0.1, cmap="viridis")
    plt.xlabel("main direction", fontdict={"size": fontsize})
    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title("Length", fontsize=fontsize + 4)

    length_visualization_save_path = os.path.join(base_path, "length_visualization.png")
    plt.savefig(length_visualization_save_path, bbox_inches="tight", pad_inches=0.1)
    plt.clf()

    plt.figure(figsize=(10, 7))

    xtarget = np.arange(min_value, max_value + 1)

    projection = latent_variable.matmul(analyzer.latent_weight).cpu().numpy()
    plt.scatter(target, projection, c=target, s=0.1)
    plt.plot(xtarget, analyzer.latent_projection_dict[min_value:].cpu().numpy())

    plt.xlabel("length", fontsize=fontsize)
    plt.ylabel("projection", fontsize=fontsize)
    plt.title("Length", fontsize=fontsize + 4)

    target_length_plot_save_path = os.path.join(base_path, "target_length_plot.png")
    plt.savefig(target_length_plot_save_path, bbox_inches="tight", pad_inches=0.1)