import os

import numpy as np
import pandas as pd
from tensorflow import keras

from generators import DataGenerator
from model.ConvModel import ConvModel

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)

    print("Creating a new model")

    # shape = (193, 81, 1)
    model: keras.models.Model = ConvModel()
    model.compile(
        # TODO Jak nadal będzie źle to zmień krok
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )
    return model


if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    *train, validation = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation}
    labels: dict = df.to_dict()['target']

    params: dict = {
        # 'dim': (193, 81),
        'dim': (64, 64),
        'batch_size': 16,
        'n_channels': 1,
        'shuffle': True
    }

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # strategy = distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    # with strategy.scope():
    # shape = (193, 81, 1)
    model = make_or_restore_model()

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=50,
        verbose=1,
        callbacks=[keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )]
    )
