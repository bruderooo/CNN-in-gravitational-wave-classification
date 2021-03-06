import os

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from generators import DataGenerator
from model import ResNet

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model(params):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)

    print("Creating a new model")

    model: keras.models.Model = ResNet()
    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall()],
    )

    model.my_summary((*params['dim'], params['n_channels']))

    return model


if __name__ == '__main__':
    df = pd.read_csv('data_spectogram_one_signal/training_labels.csv', sep=',').sample(frac=1).set_index('id')

    *train, validation = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation}
    labels: dict = df.to_dict()['target']

    params: dict = {
        'dim': (64, 64),
        'batch_size': 256,
        'n_channels': 1,
        'shuffle': True
    }

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    model = make_or_restore_model(params)

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        epochs=200,
        verbose=1,
        callbacks=[
            ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"),
            CSVLogger(filename="log.csv", separator=',', append=True)
        ]
    )
