import numpy as np
import pandas as pd

from DataGenerator import DataGenerator

if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()

    params: dict = {'dim': (32, 32, 32),
                    'batch_size': 64,
                    'n_classes': 6,
                    'n_channels': 1,
                    'shuffle': True}

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)
