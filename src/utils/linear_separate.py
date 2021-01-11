import os
import numpy as np
# from src.model.logistic_regression import LogisticRegressionClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.tsv_reader import read_field
import joblib

def linear_separate(config: dict) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    base_path = config["base_path"]
    train_sample_path = os.path.join(base_path, "vanilla_sample_train.tsv")
    dev_sample_path = os.path.join(base_path, "vanilla_sample_dev.tsv")
    test_sample_path = os.path.join(base_path, "vanilla_sample_test.tsv")
    train_latent_variable_path = os.path.join(base_path, "vanilla_sample_train.npy")
    dev_latent_variable_path = os.path.join(base_path, "vanilla_sample_dev.npy")
    test_latent_variable_path = os.path.join(base_path, "vanilla_sample_test.npy")
    train_latent_variable = np.load(train_latent_variable_path)
    dev_latent_variable = np.load(dev_latent_variable_path)
    test_latent_variable = np.load(test_latent_variable_path)
    train_label = np.asarray(read_field(train_sample_path, "label"))
    dev_label = np.asarray(read_field(dev_sample_path, "label"))
    test_label = np.asarray(read_field(test_sample_path, "label"))

    model = LogisticRegression()
    model.fit(train_latent_variable, train_label)
    test_prediction = model.predict(test_latent_variable)
    accuracy = (test_prediction == test_label).astype(np.float32).mean()
    print("category linear separate accuracy: %.4f" % accuracy)
    model_save_path = os.path.join(base_path, "classifier.pkl")
    joblib.dump(model, model_save_path)