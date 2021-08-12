import numpy as np
import pandas as pd
from tensorflow import keras

from DataGenerator import DataGenerator


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=512, kernel_size=12)(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=12)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=12)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(128, activation="relu")(gap)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(output_layer)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    # TODO to usuń xd
    df = df.head(5_000)

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()['target']

    params: dict = {'dim': (3, 4096),
                    'batch_size': 64,
                    'n_channels': 1,
                    'shuffle': True}

    training_generator: keras.utils.Sequence = DataGenerator(partition['train'], labels, **params)
    validation_generator: keras.utils.Sequence = DataGenerator(partition['validation'], labels, **params)

    model: keras.models.Model = make_model((3, 4096))
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=10,
        verbose=2,
    )
