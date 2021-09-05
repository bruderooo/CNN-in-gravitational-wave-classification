from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation


class StrideBlock(keras.layers.Layer):

    def __init__(self, filters):
        super(StrideBlock, self).__init__()

        self.conv = Conv2D(filters, (1, 1), padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)

        return x
