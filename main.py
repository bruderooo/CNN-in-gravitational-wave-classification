import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input, Dense, MaxPool2D, Flatten, Conv2D

from DataGenerator import DataGenerator


def make_model_spectogram(input_shape):
    input_layer = Input(input_shape)

    conv = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu')(input_layer)
    conv = BatchNormalization()(conv)
    conv = MaxPool2D((2, 2))(conv)

    flatt = Flatten()(conv)

    hidden_layer = Dense(64, activation="relu")(flatt)
    output_layer = Dense(1, activation="sigmoid")(hidden_layer)

    return Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    # TODO to usuń xd
    df = df.head(25_000)

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()['target']

    params: dict = {'dim': (18, 129),
                    'batch_size': 64,
                    'n_channels': 3,
                    'shuffle': True}

    training_generator: keras.utils.Sequence = DataGenerator(partition['train'], labels, **params)
    validation_generator: keras.utils.Sequence = DataGenerator(partition['validation'], labels, **params)

    model: keras.models.Model = make_model_spectogram((18, 129, 3))
    model.compile(
        # Sprawdzone 1, 0.5, 0.2, 0.01, 0.005 narazie słabiutko (za duże)
        # czas sprawdzić coś innego, zmieniłem parametry kroku
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=20000,
        verbose=2,
    )
