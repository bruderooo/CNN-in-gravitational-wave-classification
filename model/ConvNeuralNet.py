from tensorflow import keras


class ConvNeuralNet(keras.Model):

    def __init__(self):
        super(ConvNeuralNet, self).__init__()

        self.conv1 = keras.layers.Conv2D(16, (3, 3))
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = keras.layers.Conv2D(32, (3, 3))
        self.maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = keras.layers.Conv2D(64, (3, 3))
        self.maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(512, activation='relu')
        self.d2 = keras.layers.Dense(64, activation='relu')
        self.out = keras.layers.Dense(1, activation='sigmoid')

        self.dropout = keras.layers.Dropout(0.2)
        self.act = keras.layers.LeakyReLU(alpha=0.1)

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
