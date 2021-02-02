import os
from src.analyze.categorical_attribute_analyzer import CategoricalAttributeAnalyzer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hyperanalysis.visualization.lda import linear_discriminant_analysis
import pickle
from src.utils.first_upper import first_upper

def category_analyze(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    latent_size = config["text_vae"]["latent_size"]

    if "yelp" in base_path:
        attribute = "sentiment"
    elif "amazon" in base_path:
        attribute = "topic"
    else:
        raise ValueError("error")

    category_sets = {
        "sentiment": ["negative", "positive"],
        "topic": ["clothing", "game", "sports", "health"]
    }
    # topics = ["Clothing_Shoes_and_Jewelry", "Video_Games", "Sports_and_Outdoors", "Health_and_Personal_Care"]

    cmap_obj = {
        "sentiment": plt.cm.viridis,
        "topic": plt.cm.Dark2
    }

    fontsize = 26
    threshold = 0.7

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

    confident_projected_latent_variable = projected_latent_variable[probability >= threshold]
    confident_target = target[probability >= threshold]

    projected_latent_variable = projected_latent_variable.cpu().numpy()
    target = target.cpu().numpy()
    confident_projected_latent_variable = confident_projected_latent_variable.cpu().numpy()
    confident_target = confident_target.cpu().numpy()

    category_visualization_save_path = os.path.join(base_path, "category_visualization.png")
    confident_category_visualization_save_path = os.path.join(base_path, "confident_category_visualization.png")

    num_categories = len(category_sets[attribute])
    # custom_lines = [Line2D([0], [0], color=cmap_obj[attribute](0.0), lw=2),
    #                 Line2D([0], [0], color=cmap_obj[attribute](1.0), lw=2)]
    custom_lines = [
        Line2D([0], [0], color=cmap_obj[attribute](float(i) / (num_categories - 1)), lw=2) for i in range(num_categories)
    ]

    plt.legend(custom_lines, category_sets[attribute])

    plt.figure(figsize=(10, 7.5))

    plt.scatter(projected_latent_variable[:, 0], projected_latent_variable[:, 1], c=target, s=0.1, cmap="viridis")
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(custom_lines, category_sets[attribute], fontsize=fontsize)
    plt.title(first_upper(attribute), fontsize=fontsize + 4)

    plt.savefig(category_visualization_save_path, bbox_inches="tight", pad_inches=0.1)
    plt.clf()

    plt.figure(figsize=(10, 7.5))

    plt.scatter(confident_projected_latent_variable[:, 0], confident_projected_latent_variable[:, 1], c=confident_target, s=0.1, cmap="viridis")
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(custom_lines, category_sets[attribute], fontsize=fontsize)
    plt.title("%s (confidence $\geq$ %.2f)" % (first_upper(attribute), threshold), fontsize=fontsize + 4)

    plt.savefig(confident_category_visualization_save_path, bbox_inches="tight", pad_inches=0.1)