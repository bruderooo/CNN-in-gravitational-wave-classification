from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Dropout, Add, MaxPool2D

from model import ConvBlock


class ConvModelSkip(keras.Model):

    def __init__(self):
        super(ConvModelSkip, self).__init__()

        self.conv1 = ConvBlock(filters=32)
        self.conv2 = ConvBlock(filters=64)
        self.conv3 = ConvBlock(filters=128)
        self.conv4 = ConvBlock(filters=256)

        self.conv2_3_skip = Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.max_pool = MaxPool2D((4, 4))

        self.flatten = GlobalAveragePooling2D()

        self.hidden_layer1 = Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6)
        )
        self.hidden_layer2 = Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6)
        )

        self.output_layer = Dense(1, activation="sigmoid")

        self.drop = Dropout(0.5)
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        skip = self.conv2_3_skip(inputs)
        skip = self.max_pool(skip)

        x = self.add([x, skip])

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.hidden_layer1(x)
        x = self.drop(x)

        x = self.hidden_layer2(x)
        x = self.drop(x)

        x = self.output_layer(x)
        return x

    def get_config(self):
        return {}
