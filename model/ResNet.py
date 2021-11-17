from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense, \
    Add


class ResNet(keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, (7, 7), padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))

        self.id1a = IdentityBlock(64, (3, 3))
        self.id1b = IdentityBlock(64, (3, 3))

        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


class IdentityBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, inputs])
        x = self.act(x)

        return x

    def get_config(self):
        return {
            'filters': self.conv1.filters,
            'kernel_size': self.conv1.kernel_size
        }

    @classmethod
    def from_config(cls, config):
        return cls(config['filters'], config['kernel_size'])
