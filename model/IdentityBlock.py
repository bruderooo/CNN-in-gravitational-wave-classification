from keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow import keras


class IdentityBlock(keras.Model):

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.skip_conv = Conv2D(filters, kernel_size, padding='same')
        self.bn3 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        z = self.skip_conv(input_tensor)
        z = self.bn3(z)
        z = self.act(z)

        x = self.add([x, z])
        x = self.act(x)

        return x
