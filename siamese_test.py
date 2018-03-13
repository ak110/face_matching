"""Siamese Networksのテスト。"""
import pathlib

import keras
import keras.preprocessing.image
import numpy as np
import sklearn.metrics
from tqdm import tqdm

from siamese_train import load_image

BATCH_SIZE = 32

TEST_DIRS = [p for p in pathlib.Path('data/test').iterdir()]
CLASS_NAMES = [p.name for p in TEST_DIRS]
CLASS_NAMES_TO_ID = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
TRAIN_DIRS = [p for p in pathlib.Path('data/train').iterdir() if p.name in CLASS_NAMES]


def _main():
    decoder = keras.models.load_model('decoder.h5', compile=False)

    X_train, y_train = [], []
    for p in TRAIN_DIRS:
        class_id = CLASS_NAMES_TO_ID[p.name]
        for x in p.iterdir():
            X_train.append(x)
            y_train.append(class_id)

    X_test, y_test = [], []
    for p in TEST_DIRS:
        class_id = CLASS_NAMES_TO_ID[p.name]
        for x in p.iterdir():
            X_test.append(x)
            y_test.append(class_id)

    # trainのdecode
    feature_train = []
    for x in tqdm(X_train):
        img = load_image(x, train=False)
        feats = decoder.predict(np.expand_dims(img, axis=0))[0]
        feature_train.append(feats)
    feature_train = np.array(feature_train)

    # testのdecode & 予測
    true_list = []
    pred_list = []
    for x, y in tqdm(zip(X_test, y_test)):
        img = load_image(x, train=False)
        feats = decoder.predict(np.expand_dims(img, axis=0))[0]
        distances = _distance(feature_train, feats[np.newaxis, :])
        assert distances.shape == (len(y_train),)
        pred_test = y_train[distances.argmin(axis=0)]
        pred_list.append(pred_test)
        true_list.append(y)
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)

    print('test accuracy: ', sklearn.metrics.accuracy_score(true_list, pred_list))


def _distance(x0, x1):
    # 非負値のコサイン類似度：似てたら1、似てなかったら0。距離(のようなもの)にするため1から引いた値にする。
    x0 = _l2_normalize(x0, axis=-1)
    x1 = _l2_normalize(x1, axis=-1)
    d = 1 - np.sum(x0 * x1, axis=-1)
    return d


def _l2_normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis))
    return x / np.expand_dims(norm, axis=-1)


if __name__ == '__main__':
    _main()
