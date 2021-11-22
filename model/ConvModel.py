from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from model import ConvBlock


class ConvModel(keras.Model):

    def __init__(self):
        super(ConvModel, self).__init__()

        self.conv1 = ConvBlock(filters=32, kernel_size=(5, 5))
        self.conv2 = ConvBlock(filters=64)
        self.conv3 = ConvBlock(filters=128)
        self.conv4 = ConvBlock(filters=256)

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

    def get_config(self):
        return {}
