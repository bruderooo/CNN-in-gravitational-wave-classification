import numpy as np
import pandas as pd
from tensorflow import keras

from generators import DataGenerator
from model import ResNeXt
from model import ResNet

if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    # TODO to usuń xd
    df = df.head(250_000)

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()['target']

    params: dict = {'dim': (18, 129),
                    'batch_size': 128,
                    'n_channels': 3,
                    'shuffle': True}

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # shape = (18, 129, 3)
    # model: keras.models.Model = ResNeXt()
    model: keras.models.Model = ResNet()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=1000,
        verbose=2,
        callbacks=[keras.callbacks.CSVLogger(
            'log.csv', separator=',', append=False
        )]
    )
