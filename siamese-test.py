"""Siamese Networksのテスト。"""
import keras
import keras.preprocessing.image
import numpy as np
import sklearn.metrics
from tqdm import tqdm

BATCH_SIZE = 32


def _main():
    decoder = keras.models.load_model('decoder.h5', compile=False)

    idg = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    gen1 = idg.flow_from_directory('data/train', target_size=(256, 256), batch_size=BATCH_SIZE, shuffle=False)
    gen2 = idg.flow_from_directory('data/test', target_size=(256, 256), batch_size=BATCH_SIZE, shuffle=False)
    assert gen1.num_classes == gen2.num_classes

    # trainのdecode
    steps = (gen1.samples + gen1.batch_size - 1) // gen1.batch_size
    feature_list = []
    label_list = []
    for i, (X_batch, y_batch) in enumerate(tqdm(gen1, total=steps)):
        if i >= steps:
            break
        feature_list.extend(decoder.predict(X_batch))
        label_list.extend(y_batch.argmax(axis=-1))
    feature_list = np.array(feature_list)
    label_list = np.array(label_list)

    # testのdecode & 予測
    steps = (gen2.samples + gen2.batch_size - 1) // gen2.batch_size
    true_list = []
    pred_list = []
    for i, (X_batch, y_batch) in enumerate(tqdm(gen2, total=steps)):
        if i >= steps:
            break
        feats = decoder.predict(X_batch)
        true_list.extend(y_batch.argmax(axis=-1))
        for f in feats:
            rms = _distance(feature_list, f[np.newaxis, :])
            assert len(rms) == len(label_list)
            pred_list.append(label_list[rms.argmin(axis=0)])
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
