from keras.layers import GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, MaxPool2D, Flatten, Conv2D, Activation


class ConvBlock(keras.Model):
    def __init__(self, filters, kernel_size=(1, 8)):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(filters=filters, kernel_size=kernel_size)
        self.batch = BatchNormalization()
        self.pool = MaxPool2D((1, 4))
        self.act = Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch(x)
        x = self.pool(x)
        x = self.act(x)
        return x


class ConvModel(keras.Model):

    def __init__(self):
        super(ConvModel, self).__init__()

        self.act = Activation('relu')

        self.conv1 = ConvBlock(filters=32, kernel_size=(1, 16))
        self.conv2 = ConvBlock(filters=64)
        self.conv3 = ConvBlock(filters=128)
        self.conv4 = ConvBlock(filters=256)

        self.flatten = GlobalAveragePooling2D()

        self.hidden_layer1 = Dense(128)
        self.hidden_layer2 = Dense(64)
        self.output_layer = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.hidden_layer1(x)
        x = self.act(x)

        x = self.hidden_layer2(x)
        x = self.act(x)

        x = self.output_layer(x)
        return x
