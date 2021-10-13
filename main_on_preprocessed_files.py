import numpy as np
import pandas as pd
from tensorflow import keras

from generators import DataGeneratorCQT
from model import ResNet

if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    # TODO to usuń xd
    # df = df.head(50)

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()['target']

    params: dict = {'dim': (193, 81),
                    'batch_size': 10,
                    'n_channels': 1,
                    'shuffle': True}

    training_generator = DataGeneratorCQT(partition['train'], labels, **params)
    validation_generator = DataGeneratorCQT(partition['validation'], labels, **params)

    # shape = (193, 81, 1)
    model: keras.models.Model = ResNet()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    checkpoint_filepath = 'tmp/checkpoint'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=1_000,
        verbose=2,
        callbacks=[keras.callbacks.CSVLogger('tmp/log.csv', separator=',', append=False),
                   model_checkpoint_callback,
                   callback]
    )
