from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation


class IdentityBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()

        self.conv1 = Conv2D(filters, (1, 1), padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, (3, 3), padding='same')
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(kernel_size, (1, 1), padding='same')
        self.bn3 = BatchNormalization()

        self.act = Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        return x