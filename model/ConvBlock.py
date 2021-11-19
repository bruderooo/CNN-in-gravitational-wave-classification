from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, Conv2D, Activation


class ConvBlock(keras.Model):
    def __init__(self, filters, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.batch = BatchNormalization()
        self.pool = MaxPool2D((2, 2))
        self.activation = Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

    def get_config(self):
        return {
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size
        }
