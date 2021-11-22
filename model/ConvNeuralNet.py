from tensorflow import keras


class ConvNeuralNet(keras.Model):

    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.padding = keras.layers.ZeroPadding2D(padding=((0, 2), (0, 2)))

        self.conv1a = keras.layers.Conv2D(64, (3, 3), padding='valid')
        self.conv1b = keras.layers.Conv2D(128, (3, 3), padding='same')
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2a = keras.layers.Conv2D(128, (3, 3), padding='same')
        self.conv2b = keras.layers.Conv2D(256, (3, 3), padding='same')
        self.maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128)
        self.d2 = keras.layers.Dense(1, activation='sigmoid')

        self.dropout = keras.layers.Dropout(0.5)
        self.act = keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs):
        x = self.padding(inputs)
        x = self.conv1a(x)
        x = self.act(x)

        x = self.conv1b(x)
        x = self.act(x)

        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.act(x)

        x = self.conv2b(x)
        x = self.act(x)

        x = self.maxpool2(x)

        x = self.flatten(x)
        # x = self.d1(x)
        # x = self.act(x)
        #
        # x = self.dropout(x)
        x = self.d2(x)
        return x
