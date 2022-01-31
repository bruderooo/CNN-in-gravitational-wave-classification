import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model

from generators.DataGeneratorTesting import DataGeneratorTesting

if __name__ == '__main__':
    df = pd.read_csv('data_spectogram_one_signal/sample_submission.csv', sep=',')

    params: dict = {
        'dim': (64, 64),
        'batch_size': 256,
        'n_channels': 1
    }

    models_dict = {
        '1': 'rdy_models/d1_256_drop_20%/ckpt-50',
        '2': 'rdy_models/resnet_2_bloki/ckpt-250'
    }

    df_cnn = df.copy()
    test_generator1 = DataGeneratorTesting(df['id'].to_numpy(), **params)
    model1: keras.Model = load_model(models_dict['1'])
    df_cnn['target'] = model1.predict(test_generator1)
    df_cnn.to_csv('submission_cnn.csv')

    df_resnet = df.copy()
    test_generator2 = DataGeneratorTesting(df['id'].to_numpy(), **params)
    model2: keras.Model = load_model(models_dict['2'])
    df_resnet['target'] = model2.predict(test_generator2)
    df_resnet.to_csv('submission_resnet.csv')
