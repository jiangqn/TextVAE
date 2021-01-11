import os
from src.analyze.classification_analyzer import ClassificationAnalyzer
import matplotlib.pyplot as plt
from hyperanalysis.visualization.lda import linear_discriminant_analysis
from hyperanalysis.visualization.lra import linear_regression_analysis

def category_analyze(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    latent_size = config["text_vae"]["latent_size"]

    analyzer = ClassificationAnalyzer(
        base_path=base_path,
        target_name="label",
        hidden_size=latent_size,
        n_blocks=1
    )

    analyzer.fit(num_epoches=10)
    # analyzer.load_model()

    latent_variable, transformed_latent_variable, target = analyzer.get_data()

    projected_latent_variable = linear_discriminant_analysis(latent_variable, target).cpu().numpy()
    projected_transformed_latent_variable = linear_discriminant_analysis(transformed_latent_variable,
                                                                       target).cpu().numpy()
    target = target.cpu().numpy()

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)

    plt.scatter(projected_latent_variable[:, 0], projected_latent_variable[:, 1], c=target, s=0.1, cmap="viridis")
    plt.colorbar()
    plt.title("latent space")

    plt.subplot(1, 2, 2)
    plt.scatter(projected_transformed_latent_variable[:, 0], projected_transformed_latent_variable[:, 1], c=target,
                s=0.1, cmap="viridis")
    plt.colorbar()
    plt.title("transformed latent space")

    figure_save_path = os.path.join(base_path, "category_visualization.png")
    plt.savefig(figure_save_path)