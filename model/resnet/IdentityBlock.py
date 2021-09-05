from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add


class IdentityBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.add([x, inputs])
        x = self.act(x)

        return x
