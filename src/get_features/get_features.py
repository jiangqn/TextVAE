import pandas as pd
import os
import math
from src.get_features.get_category import get_categorical_features_from_tsv
from src.get_features.get_ppl import get_ppl_from_tsv
from src.get_features.get_length import get_length
from src.get_features.get_depth import get_depth
from src.utils.tsv_reader import read_field

def get_features(config: dict) -> None:

    base_path = config["base_path"]

    for division in ["train", "dev", "test"]:

        vanilla_sample_save_path = os.path.join(base_path, "vanilla_sample_%s.tsv" % division)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

        vocab_path = os.path.join(base_path, "vocab.pkl")
        text_cnn_path = os.path.join(base_path, "text_cnn.pkl")
        language_model_path = os.path.join(base_path, "language_model.pkl")

        df = pd.read_csv(vanilla_sample_save_path, delimiter="\t")
        sentences = list(df["sentence"])

        if "label" in config["features"]:
            print("get categorical features")
            label = get_categorical_features_from_tsv(file_path=vanilla_sample_save_path, batch_size=config["text_cnn"]["batch_size"],
                model_path=text_cnn_path, vocab_path=vocab_path, output_category=True)
            df["label"] = label

        # print("get ppl")
        # ppl = get_ppl_from_tsv(file_path=sample_save_path, batch_size=config["language_model"]["batch_size"],
        #     model_path=language_model_path, vocab_path=vocab_path)

        if "length" in config["features"]:
            print("get length")
            length = get_length(sentences)
            df["length"] = length

        if "depth" in config["features"]:
            print("get depth")
            depth = get_depth(sentences, processes=20)
            df["depth"] = depth

        df.to_csv(vanilla_sample_save_path, sep="\t")