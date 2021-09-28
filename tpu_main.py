import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import distribute
from tensorflow import keras

from generators import DataGenerator
from model import ResNet

if __name__ == '__main__':
    # detect and init the TPU
    tpu = distribute.cluster_resolver.TPUClusterResolver.connect()

    # instantiate a distribution strategy
    tpu_strategy = distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync

    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # On Kaggle you can also use KaggleDatasets().get_gcs_path() to obtain the GCS path of a Kaggle dataset
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    # TODO to usuń xd
    df = df.head(100_000)

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()['target']

    params: dict = {'dim': (18, 129),
                    'batch_size': BATCH_SIZE,
                    'n_channels': 3,
                    'shuffle': False}

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    filenames = tf.io.gfile.glob("gs://flowers-public/tfrecords-jpeg-512x512/*.tfrec")
    dataset = tf.data.TFRecordDataset.from_generator(generator=training_generator)
    dataset = dataset.with_options(ignore_order)

    with tpu_strategy.scope():
        model: keras.models.Model = ResNet()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
            steps_per_execution=32
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
