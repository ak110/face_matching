"""Siamese Networksの学習。"""
import pathlib

import keras
import keras.backend as K
import keras.preprocessing.image
import numpy as np

BATCH_SIZE = 32

TRAIN_DIRS = [p for p in pathlib.Path('data/train').iterdir()]
TEST_DIRS = [p for p in pathlib.Path('data/test').iterdir()]
# 2枚以上の画像を持つデータ一覧
TRAINABLE_DIRS = [p for p in pathlib.Path('data/train').iterdir() if len(list(p.iterdir())) >= 2]


def _main():
    print('len(TRAINABLE_DIRS) = ', len(TRAINABLE_DIRS))

    x = inp = keras.layers.Input((256, 256, 3))
    x = _conv_bn_act(32, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = _conv_bn_act(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = _conv_bn_act(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = _conv_bn_act(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = _conv_bn_act(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = _conv_bn_act(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = _conv_bn_act(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = _conv_bn_act(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = _conv_bn_act(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='sigmoid')(x)
    decoder = keras.models.Model(inp, x)
    decoder.summary()

    inp1 = keras.layers.Input((256, 256, 3))
    inp2 = keras.layers.Input((256, 256, 3))
    d1 = decoder(inp1)
    d2 = decoder(inp2)
    x = keras.layers.Lambda(_distance, _distance_shape)([d1, d2])
    siamese = keras.models.Model([inp1, inp2], x)
    siamese.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.SGD(momentum=0.9, nesterov=True))
    siamese.summary()

    base_lr = 0.5 * BATCH_SIZE / 256
    main_epochs = 20
    lr_list = [base_lr] * main_epochs + [base_lr / 10] * (main_epochs // 2) + [base_lr / 100] * (main_epochs // 2)

    callbacks = []
    callbacks.append(keras.callbacks.LearningRateScheduler(lambda ep: lr_list[ep]))
    callbacks.append(keras.callbacks.CSVLogger('siamese-train.tsv', separator='\t'))
    siamese.fit_generator(
        generator=_gen_samples(BATCH_SIZE, train=True),
        steps_per_epoch=512,
        validation_data=_gen_samples(BATCH_SIZE, train=False),
        validation_steps=32,
        class_weight=np.array([0.2, 1.8]),
        epochs=len(lr_list),
        callbacks=callbacks)

    decoder.save('decoder.h5')


def _conv_bn_act(*args, **kwargs):
    def _layer(x):
        x = keras.layers.Conv2D(*args, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x
    return _layer


def _distance(x):
    # 非負値のコサイン類似度：似てたら1、似てなかったら0。距離(のようなもの)にするため1から引いた値にする。
    x0 = K.l2_normalize(x[0], axis=-1)
    x1 = K.l2_normalize(x[1], axis=-1)
    d = 1 - K.sum(x0 * x1, axis=-1)
    d = K.expand_dims(d, axis=-1)
    return d


def _distance_shape(input_shape):
    assert input_shape[0] == input_shape[1]
    return input_shape[0][:len(input_shape[0]) - 1] + (1,)


def _gen_samples(batch_size, train):
    while True:
        y = np.random.choice([0] + [1], batch_size)  # 一致：不一致
        X1 = np.empty((batch_size, 256, 256, 3))
        X2 = np.empty((batch_size, 256, 256, 3))
        for i in range(batch_size):
            if train:
                if y[i] == 0:  # 一致
                    pair = np.random.choice(list(np.random.choice(TRAINABLE_DIRS).iterdir()), 2, replace=False)
                    X1[i] = _load_image(pair[0], train=train)
                    X2[i] = _load_image(pair[1], train=train)
                else:
                    assert y[i] == 1  # 不一致
                    p1, p2 = np.random.choice(TRAIN_DIRS, 2, replace=False)
                    X1[i] = _load_image(np.random.choice(list(p1.iterdir())), train=train)
                    X2[i] = _load_image(np.random.choice(list(p2.iterdir())), train=train)
            else:
                p1 = np.random.choice(TEST_DIRS)
                if y[i] == 0:
                    p2 = [d for d in TRAIN_DIRS if d.name == p1.name][0]  # 一致
                else:
                    p2 = np.random.choice([d for d in TRAIN_DIRS if d.name != p1.name])  # 不一致
                X1[i] = _load_image(np.random.choice(list(p1.iterdir())), train=train)
                X2[i] = _load_image(np.random.choice(list(p2.iterdir())), train=train)
        yield [X1, X2], y


def _load_image(path, train):
    img = keras.preprocessing.image.load_img(path, target_size=(256, 256))

    # Data Augmentation
    if train:
        if np.random.rand() < 0.5:
            img = img.rotate(np.random.uniform(-10, +10))
        if np.random.rand() < 0.5:
            import PIL.ImageOps
            img = PIL.ImageOps.mirror(img)

    x = keras.preprocessing.image.img_to_array(img)
    x /= 255.  # rescale=1. / 255
    return x


if __name__ == '__main__':
    _main()
