"""通常の分類。"""
import keras
import keras.preprocessing.image

BATCH_SIZE = 32


def _main():
    idg1 = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        rescale=1. / 255)
    idg2 = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    gen1 = idg1.flow_from_directory('data/train', target_size=(256, 256), batch_size=BATCH_SIZE)
    gen2 = idg2.flow_from_directory('data/test', target_size=(256, 256), batch_size=BATCH_SIZE, shuffle=False)
    assert gen1.num_classes == gen2.num_classes
    num_classes = gen1.num_classes

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
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inp, x)
    model.summary()

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=1e-3 * BATCH_SIZE / 128),
        metrics=['accuracy']
    )

    model.fit_generator(
        gen1,
        steps_per_epoch=gen1.samples // gen1.batch_size,
        epochs=30,
        validation_data=gen2,
        validation_steps=gen2.samples // gen2.batch_size)


def _conv_bn_act(*args, **kwargs):
    def _layer(x):
        x = keras.layers.Conv2D(*args, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x
    return _layer


if __name__ == '__main__':
    _main()
