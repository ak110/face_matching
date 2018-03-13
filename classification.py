"""通常の分類。"""
import keras
import keras.preprocessing.image

BATCH_SIZE = 16


def _main():
    idg1 = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        rescale=1. / 255)
    idg2 = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    gen1 = idg1.flow_from_directory('data/train_only', target_size=(224, 224), batch_size=BATCH_SIZE)
    gen2 = idg2.flow_from_directory('data/test', target_size=(224, 224), batch_size=BATCH_SIZE, shuffle=False)
    assert gen1.num_classes == gen2.num_classes
    num_classes = gen1.num_classes

    base_model = keras.applications.Xception(include_top=False)
    for layer in base_model.layers:
        if layer.name == 'block14_sepconv1':
            break
        elif not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    x = base_model.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='zeros')(x)
    model = keras.models.Model(base_model.inputs, x)
    model.summary()

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(momentum=0.9, nesterov=True),
        metrics=['acc'])

    base_lr = 1e-1
    main_epochs = 20
    lr_list = [base_lr] * main_epochs + [base_lr / 10] * (main_epochs // 2) + [base_lr / 100] * (main_epochs // 2)

    callbacks = []
    callbacks.append(keras.callbacks.LearningRateScheduler(lambda ep: lr_list[ep]))
    callbacks.append(keras.callbacks.CSVLogger('classification_history.tsv', separator='\t'))
    model.fit_generator(
        generator=gen1,
        steps_per_epoch=gen1.samples // gen1.batch_size,
        epochs=len(lr_list),
        validation_data=gen2,
        validation_steps=gen2.samples // gen2.batch_size,
        callbacks=callbacks)


def _conv_bn_act(*args, **kwargs):
    def _layer(x):
        x = keras.layers.Conv2D(*args, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x
    return _layer


if __name__ == '__main__':
    _main()
