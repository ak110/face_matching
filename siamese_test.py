"""Siamese Networksのテスト。"""
import logging
import pathlib

import keras
import keras.preprocessing.image
import numpy as np
import sklearn.metrics
from tqdm import tqdm

from siamese_train import distance_np, load_image

BATCH_SIZE = 32

TEST_DIRS = [p for p in pathlib.Path('data/test').iterdir()]
CLASS_NAMES = [p.name for p in TEST_DIRS]
CLASS_NAMES_TO_ID = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
TRAIN_DIRS = [p for p in pathlib.Path('data/train').iterdir() if p.name in CLASS_NAMES]
assert len(TRAIN_DIRS) == len(TEST_DIRS)

try:
    import better_exceptions
except BaseException:
    pass


def _main():
    decoder = keras.models.load_model('decoder.h5', compile=False)

    logging.basicConfig(level=logging.INFO, filename='siamese_result.txt', filemode='w')
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    X_train, y_train = [], []
    for p in TRAIN_DIRS:
        class_id = CLASS_NAMES_TO_ID[p.name]
        x_class = [x for x in p.iterdir()]
        assert len(x_class) >= 1
        X_train.extend(x_class)
        y_train.extend([class_id] * len(x_class))
    y_train = np.array(y_train)

    X_test, y_test = [], []
    for p in TEST_DIRS:
        class_id = CLASS_NAMES_TO_ID[p.name]
        x_class = [x for x in p.iterdir()]
        assert len(x_class) >= 1
        X_test.extend(x_class)
        y_test.extend([class_id] * len(x_class))
        assert class_id in y_train
    y_test = np.array(y_test)

    # trainのdecode
    feature_train = []
    for X_batch in tqdm(np.array_split(X_train, len(X_train) // BATCH_SIZE), ascii=True):
        imgs = np.array([load_image(x, train=False) for x in X_batch])
        feats = decoder.predict(imgs)
        feature_train.extend(feats)
    feature_train = np.array(feature_train)
    assert len(feature_train) == len(y_train)

    # testのdecode & 予測
    true_list = []
    pred_list = []
    order_list = []
    match_dist_info = []
    unmatch_dist_info = []
    for x, y in tqdm(list(zip(X_test, y_test)), ascii=True):
        img = load_image(x, train=False)
        feats = decoder.predict(np.expand_dims(img, axis=0))[0]
        distances = distance_np(feature_train, feats[np.newaxis, :])
        assert distances.shape == (len(y_train),)
        pred_test = y_train[distances.argmin(axis=0)]
        pred_list.append(pred_test)
        true_list.append(y)
        for i, j in enumerate(distances.argsort(axis=0)):
            if y == y_train[j]:
                order_list.append(i)
                break

        def _get_info(a):
            assert len(a) > 0
            return np.amin(a), np.amax(a), np.mean(a), np.median(a)
        match_dist_info.append(_get_info(distances[y_train == y]))
        unmatch_dist_info.append(_get_info(distances[y_train != y]))
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)
    order_list = np.array(order_list)

    logger.info('mean order:      %.1f', np.mean(order_list))
    logger.info('test accuracy:   %.4f', sklearn.metrics.accuracy_score(true_list, pred_list))
    # logger.info('top 1 accuracy:  %.4f', np.mean(order_list < 1))
    logger.info('top 5 accuracy:  %.4f', np.mean(order_list < 5))
    logger.info('top 10 accuracy: %.4f', np.mean(order_list < 10))
    logger.info('top 15 accuracy: %.4f', np.mean(order_list < 15))

    def _print_info(name, aa):
        logger.info(name + ':')
        logger.info('  min:    %.4f', np.mean([a[0] for a in aa]))
        logger.info('  max:    %.4f', np.mean([a[1] for a in aa]))
        logger.info('  mean:   %.4f', np.mean([a[2] for a in aa]))
        logger.info('  median: %.4f', np.mean([a[3] for a in aa]))

    _print_info('match', match_dist_info)
    _print_info('unmatch', unmatch_dist_info)


if __name__ == '__main__':
    _main()
