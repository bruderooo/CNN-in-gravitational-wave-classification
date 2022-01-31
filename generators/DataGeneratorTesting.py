import numpy as np
from tensorflow import keras


class DataGeneratorTesting(keras.utils.Sequence):

    def __init__(
            self,
            list_IDs,
            batch_size=256,
            dim=(4096,),
            n_channels=1
    ):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))

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
        X = self.__data_generation(list_id_temp)

        return X

    def __data_generation(
            self,
            list_IDs_temp
    ):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(f"data_spectogram_one_signal/test/{ID[0]}/{ID[1]}/{ID[2]}/{ID}.npy")
            X[i,] = (x - np.mean(x)) / np.max(x)

        return X
