import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data\\training_labels.csv', sep=',').sample(frac=1).set_index('id')

    *train, validation, test = np.split(df.index.values, 5)
    partition: dict = {'train': np.array(train).flatten(), 'validation': validation, 'test': test}
    labels: dict = df.to_dict()
