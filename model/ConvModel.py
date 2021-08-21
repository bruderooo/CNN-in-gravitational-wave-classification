from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, MaxPool2D, Flatten, Conv2D, Activation


class ConvModel(keras.Model):

    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = Conv2D(filters=16, kernel_size=(3, 3), padding="same")
        self.bn = BatchNormalization()
        self.max_pool = MaxPool2D((2, 2))

        self.flatten = Flatten()

        self.hidden_layer = Dense(64)
        self.output_layer = Dense(1, activation="sigmoid")

        self.act = Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.act(x)
        x = self.bn(x)
        x = self.max_pool(x)

        x = self.max_pool(x)

        x = self.hidden_layer(x)
        x = self.act(x)

        x = self.output_layer(x)
        return x
