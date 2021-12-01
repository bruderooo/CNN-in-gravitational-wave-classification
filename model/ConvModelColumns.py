from tensorflow import keras
from tensorflow.keras import layers

from model import Model


class ConvModelColumns(Model):

    def __init__(self):
        super(ConvModelColumns, self).__init__()

        self.conv1a = layers.Conv2D(16, (1, 3))
        self.maxpool1a = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv1b = layers.Conv2D(16, (3, 1))
        self.maxpool1b = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = layers.Conv2D(32, (3, 3))
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = keras.layers.Conv2D(64, (3, 3))
        self.maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.drop1 = layers.Dropout(0.4)

        self.d2 = layers.Dense(64, activation='relu')
        self.drop2 = layers.Dropout(0.3)

        self.out = layers.Dense(1, activation='sigmoid')

        self.act = layers.LeakyReLU(alpha=0.1)

    def call(self, inputs):
        x = self.conv1a(inputs)
        x = self.act(x)
        x = self.maxpool1a(x)

        x = self.conv1b(x)
        x = self.act(x)
        x = self.maxpool1b(x)

        x = keras.layers.concatenate([x, inputs])

        x = self.conv2(x)
        x = self.act(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.drop1(x)

        x = self.d2(x)
        x = self.drop2(x)

        x = self.out(x)
        return x

    def get_config(self):
        return super(ConvModelColumns, self).get_config()
