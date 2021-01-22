import os
from src.analyze.categorical_attribute_analyzer import CategoricalAttributeAnalyzer
import matplotlib.pyplot as plt
from hyperanalysis.visualization.lda import linear_discriminant_analysis
import pickle

def category_analyze(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    latent_size = config["text_vae"]["latent_size"]

    analyzer = CategoricalAttributeAnalyzer(
        base_path=base_path,
        latent_size=latent_size
    )

    analyzer.fit()
    analyzer_path = os.path.join(base_path, "category_analyzer.pkl")
    with open(analyzer_path, "wb") as f:
        pickle.dump(analyzer, f)

    latent_variable, target, probability = analyzer.get_data(output_probability=True)

    projected_latent_variable = linear_discriminant_analysis(latent_variable, target)

    threshold = 0.95
    confident_projected_latent_variable = projected_latent_variable[probability >= threshold]
    confident_target = target[probability >= threshold]

    projected_latent_variable = projected_latent_variable.cpu().numpy()
    target = target.cpu().numpy()
    confident_projected_latent_variable = confident_projected_latent_variable.cpu().numpy()
    confident_target = confident_target.cpu.numpy()

    category_visualization_save_path = os.path.join(base_path, "category_visualization.png")
    confident_category_visualization_save_path = os.path.join(base_path, "confident_category")

    plt.figure(figsize=(10, 7))

    plt.scatter(projected_latent_variable[:, 0], projected_latent_variable[:, 1], c=target, s=0.1, cmap="viridis")
    plt.colorbar()
    plt.title("latent space")

    plt.savefig(category_visualization_save_path, bbox_inches="tight", pad_inches=0.1)
    plt.clf()

    plt.figure(figsize=(10, 7))

    plt.scatter(confident_projected_latent_variable[:, 0], confident_projected_latent_variable[:, 1], c=confident_target, s=0.1, cmap="viridis")
    plt.colorbar()
    plt.title("latent space with confidence")

    plt.savefig(confident_category_visualization_save_path, bbox_inches="tight", pad_inches=0.1)