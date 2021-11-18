from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, Conv2D


class ConvBlock(keras.Model):
    def __init__(self, filters, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')
        self.batch = BatchNormalization()
        self.pool = MaxPool2D((2, 2))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.pool(x)
        return x

    def get_config(self):
        return {
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'max_pool_size': self.pool.pool_size
        }
