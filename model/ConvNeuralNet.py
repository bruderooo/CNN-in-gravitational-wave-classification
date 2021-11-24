from tensorflow import keras
from tensorflow.keras import layers

from model import Model


class ConvNeuralNet(Model):

    def __init__(self):
        super(ConvNeuralNet, self).__init__()

        self.conv1 = layers.Conv2D(16, (3, 3))
        self.maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = layers.Conv2D(32, (3, 3))
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = keras.layers.Conv2D(64, (3, 3))
        self.maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid')

        self.dropout = layers.Dropout(0.2)
        self.act = layers.LeakyReLU(alpha=0.1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.dropout(x)

        x = self.d2(x)
        x = self.dropout(x)

        x = self.out(x)
        return x

    def my_summary(self, input_shape):
        x = layers.Input(shape=input_shape)
        return keras.Model(inputs=x, outputs=self.call(x)).summary()
