from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, MaxPool2D, GlobalAveragePooling2D, Conv2D, Activation, \
    Dropout


class ConvBlock(keras.Model):
    def __init__(self, filters, kernel_size=(4, 4)):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(filters=filters, kernel_size=kernel_size)
        self.batch = BatchNormalization()
        self.pool = MaxPool2D((2, 2))
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

        self.conv1 = ConvBlock(filters=32, kernel_size=(7, 7))
        self.conv2 = ConvBlock(filters=64)
        self.conv3 = ConvBlock(filters=128)
        self.conv4 = ConvBlock(filters=256)

        self.flatten = GlobalAveragePooling2D()

        self.hidden_layer1 = Dense(128, activation='relu')
        self.hidden_layer2 = Dense(64, activation='relu')

        self.output_layer = Dense(1, activation="sigmoid")

        self.drop = Dropout(0.2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.hidden_layer1(x)
        x = self.drop(x)

        x = self.hidden_layer2(x)
        x = self.drop(x)

        x = self.output_layer(x)
        return x
