import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

def linear_separate(config):

    base_path = config['base_path']
    # sample_save_path = input('save path: ')
    sample_save_path = os.path.join(base_path, 'sample10000.tsv')
    encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
    model_save_path = os.path.join(base_path, 'sentiment_lr.pkl')

    encoding = np.load(encoding_save_path)
    df = pd.read_csv(sample_save_path, delimiter='\t')

    sentiment = np.asarray(df['sentiment']).astype(np.float32)
    sentiment = (sentiment >= 0.5).astype(np.int32)
    model = LogisticRegression()
    # model = LinearSVC(max_iter=10000)
    model.fit(encoding, sentiment)
    pred_sentiment = model.predict(encoding)
    accuracy = (sentiment == pred_sentiment).astype(np.float32).mean()
    print('sentiment: %.4f' % accuracy)
    joblib.dump(model, model_save_path)

    for prop_name in ['length', 'depth']:
        prop = np.asarray(df[prop_name]).astype(np.int32)
        model = LogisticRegression(max_iter=500)
        model.fit(encoding, prop)
        pred_prop = model.predict(encoding)
        accuracy = (prop == pred_prop).astype(np.float32).mean()
        print('%s: %.4f' % (prop_name, accuracy))