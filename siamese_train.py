"""Siamese Networksの学習。"""
import pathlib

import keras
import keras.backend as K
import keras.preprocessing.image
import numpy as np

BATCH_SIZE = 16

TRAIN_DIRS = [p for p in pathlib.Path('data/train').iterdir()]
TEST_DIRS = [p for p in pathlib.Path('data/test').iterdir()]
# 2枚以上の画像を持つデータ一覧
TRAIN_PAIR_DIRS = [p for p in pathlib.Path('data/train').iterdir() if len(list(p.iterdir())) >= 2]


def _main():
    base_model = keras.applications.Xception(include_top=False, input_shape=(None, None, 3))
    for layer in base_model.layers:
        if layer.name == 'block14_sepconv1':
            break
        elif not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    x = base_model.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128)(x)
    decoder = keras.models.Model(base_model.inputs, x)
    decoder.summary()

    inp1 = keras.layers.Input((None, None, 3))
    inp2 = keras.layers.Input((None, None, 3))
    d1 = decoder(inp1)
    d2 = decoder(inp2)
    x = keras.layers.Lambda(_distance, _distance_shape)([d1, d2])
    siamese = keras.models.Model([inp1, inp2], x)
    siamese.compile(
        # loss=keras.losses.binary_crossentropy,
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.SGD(momentum=0.9, nesterov=True),
        metrics=['acc'])
    siamese.summary()

    print('len(TRAIN_PAIR_DIRS) = ', len(TRAIN_PAIR_DIRS))

    base_lr = 1e-1
    main_epochs = 20
    lr_list = [base_lr] * main_epochs + [base_lr / 10] * (main_epochs // 2) + [base_lr / 100] * (main_epochs // 2)

    callbacks = []
    callbacks.append(keras.callbacks.LearningRateScheduler(lambda ep: lr_list[ep]))
    callbacks.append(keras.callbacks.CSVLogger('siamese_history.tsv', separator='\t'))
    siamese.fit_generator(
        generator=_gen_samples(BATCH_SIZE, train=True),
        steps_per_epoch=512,
        validation_data=_gen_samples(BATCH_SIZE, train=False),
        validation_steps=32,
        # class_weight=np.array([4.0, 0.5]),
        epochs=len(lr_list),
        callbacks=callbacks,
        workers=8)

    decoder.save('decoder.h5')


def _conv_bn_act(*args, **kwargs):
    def _layer(x):
        x = keras.layers.Conv2D(*args, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x
    return _layer


def _distance(x):
    d = K.sqrt(K.mean(K.square(x[0] - x[1]), axis=-1))
    d = K.expand_dims(d, axis=-1)
    return d


def distance_np(x0, x1):
    """距離(numpy版)"""
    d = np.sqrt(np.mean(np.square(x0 - x1), axis=-1))
    return d


def _distance_shape(input_shape):
    assert input_shape[0] == input_shape[1]
    return input_shape[0][:len(input_shape[0]) - 1] + (1,)


def _gen_samples(batch_size, train):
    while True:
        y = []
        X1 = []
        X2 = []
        for _ in range(batch_size // (4 if train else 2)):
            if train:
                # 一致
                pair = np.random.choice(list(np.random.choice(TRAIN_PAIR_DIRS).iterdir()), 2, replace=False)
                X1.append(load_image(pair[0], train=train))
                X2.append(load_image(pair[1], train=train))
                y.append(0)
                # 不一致
                for pa in pair:
                    p2 = np.random.choice([d for d in TRAIN_DIRS if d.name != pa.name])
                    X1.append(load_image(pa, train=train))
                    X2.append(load_image(np.random.choice(list(p2.iterdir())), train=train))
                    y.append(1)
                # もう一つ不一致
                p1, p2 = np.random.choice(TRAIN_DIRS, 2, replace=False)
                X1.append(load_image(np.random.choice(list(p1.iterdir())), train=train))
                X2.append(load_image(np.random.choice(list(p2.iterdir())), train=train))
                y.append(1)
            else:
                p1 = np.random.choice(TEST_DIRS)
                p2 = [d for d in TRAIN_DIRS if d.name == p1.name][0]  # 一致
                p3 = np.random.choice([d for d in TRAIN_DIRS if d.name != p1.name])  # 不一致
                X1.append(load_image(np.random.choice(list(p1.iterdir())), train=train))
                X2.append(load_image(np.random.choice(list(p2.iterdir())), train=train))
                y.append(0)
                X1.append(X1[-1])
                X2.append(load_image(np.random.choice(list(p3.iterdir())), train=train))
                y.append(1)
        yield [np.array(X1), np.array(X2)], np.array(y)


def load_image(path, train):
    """画像の読み込み。"""
    img = keras.preprocessing.image.load_img(path, target_size=(224, 224))

    # Data Augmentation
    if train:
        if np.random.rand() < 0.5:
            img = img.rotate(np.random.uniform(-10, +10))
        if np.random.rand() < 0.5:
            import PIL.ImageOps
            img = PIL.ImageOps.mirror(img)

    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.xception.preprocess_input(x)
    x = np.squeeze(x, axis=0)
    return x


if __name__ == '__main__':
    _main()
