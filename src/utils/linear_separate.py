import os
import numpy as np
from src.model.logistic_regression import LogisticRegressionClassifier
from src.model.linear_svm import LinearSVMClassifier
from src.utils.tsv_reader import read_field
import joblib

def linear_separate(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    vanilla_sample_num = config['vanilla_sample']['sample_num']
    vanilla_sample_save_path = os.path.join(base_path, 'vanilla_sample_%d.tsv' % vanilla_sample_num)
    encoding_save_path = '.'.join(vanilla_sample_save_path.split('.')[0:-1]) + '.npy'
    model_save_path = os.path.join(base_path, 'classifier.pkl')

    encoding = np.load(encoding_save_path)

    label = np.asarray(read_field(vanilla_sample_save_path, 'label'))
    model = LogisticRegressionClassifier(max_epoches=100)
    model.fit(encoding, label)
    predition = model.predict(encoding)
    accuracy = (predition == label).astype(np.float32).mean()
    print('category linear separate accuracy: %.4f' % accuracy)
    joblib.dump(model, model_save_path)