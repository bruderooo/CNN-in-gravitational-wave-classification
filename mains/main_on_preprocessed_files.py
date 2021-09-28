import numpy as np
import pandas as pd
from tensorflow import keras

from generators import DataGeneratorCQT
from model import ResNet

if __name__ == '__main__':
    df = pd.read_csv('..\\data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    # TODO to usuń xd
    df = df.head(10_000)

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()['target']

    params: dict = {'dim': (81, 193),
                    'batch_size': 32,
                    'n_channels': 1,
                    'shuffle': True}

    training_generator = DataGeneratorCQT(partition['train'], labels, **params)
    validation_generator = DataGeneratorCQT(partition['validation'], labels, **params)

    # shape = (1, 81, 193)
    model: keras.models.Model = ResNet()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-10),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=5000,
        verbose=2,
        callbacks=[keras.callbacks.CSVLogger(
            'log.csv', separator=',', append=False
        )]
    )
