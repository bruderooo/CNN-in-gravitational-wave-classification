import numpy as np
import scipy.signal
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            list_IDs,
            labels,
            batch_size=256,
            dim=(4096,),
            n_channels=3,
            shuffle=True
    ):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(
            self,
            index
    ):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_id_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_id_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(
            self,
            list_IDs_temp
    ):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(f"data\\train\\{ID[0]}\\{ID[1]}\\{ID[2]}\\{ID}.npy")

            multiplier = [
                9.044647242705657e-20,
                8.374226193192353e-20,
                2.202426281781826e-20
            ]

            mean_ = [
                5.364163252884116e-27,
                1.215962445189085e-25,
                2.3707386590609453e-27
            ]

            tmp = []
            for sig, minus, tims in zip(x, mean_, multiplier):
                tmp.append(scipy.signal.spectrogram((sig - minus) / tims, fs=2048)[2])

            X[i,] = np.array(tmp).T

            # Store class
            y[i] = self.labels[ID]

        return X, y
