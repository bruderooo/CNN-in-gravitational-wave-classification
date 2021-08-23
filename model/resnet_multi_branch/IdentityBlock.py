from keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow import keras


class IdentityBlock(keras.Model):

    def __init__(self, filters, input_size):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = Conv2D(filters, 1, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, 1, padding='same')
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(input_size, 1, padding='same')
        self.bn3 = BatchNormalization()

        self.act = Activation('relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        return x
