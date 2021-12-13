from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, GlobalAveragePooling2D, Dense, \
    Add, Dropout

from model import Model


class ResNet(Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = Conv2D(16, (7, 7), padding='same')
        self.act = layers.LeakyReLU(alpha=0.1)
        self.max_pool = MaxPool2D((3, 3))

        self.id1a = IdentityBlock(16, (3, 3))
        self.id1b = IdentityBlock(16, (3, 3))

        self.bottleneck1 = BottleneckBlock(32, (3, 3))
        self.id2a = IdentityBlock(32, (3, 3))
        self.id2b = IdentityBlock(32, (3, 3))
        
        self.bottleneck2 = BottleneckBlock(64, (3, 3))
        self.id3a = IdentityBlock(64, (3, 3))
        self.id3b = IdentityBlock(64, (3, 3))

        self.global_pool = GlobalAveragePooling2D()
        self.drop = Dropout(0.2)

        self.hidden_layer1 = Dense(128, activation='relu')
        self.hidden_layer2 = Dense(64, activation='relu')

        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.bottleneck1(x)
        x = self.id2a(x)
        x = self.id2b(x)
        
        x = self.bottleneck2(x)
        x = self.id3a(x)
        x = self.id3b(x)

        x = self.global_pool(x)

        x = self.hidden_layer1(x)
        x = self.drop(x)

        x = self.hidden_layer2(x)
        x = self.drop(x)

        x = self.classifier(x)
        return x

    def get_config(self):
        return super(ResNet, self).get_config()


class IdentityBlock(Model):

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.conv2 = Conv2D(filters, kernel_size, padding='same')

        self.act = layers.LeakyReLU(alpha=0.1)
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.add([x, inputs])
        x = self.act(x)

        return x

    def get_config(self):
        config = super(IdentityBlock, self).get_config()
        config.update({
            'filters': self.conv1.filters,
            'kernel_size': self.conv1.kernel_size
        })
        return config


class BottleneckBlock(Model):

    def __init__(self, filters, kernel_size):
        super(BottleneckBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.conv2 = Conv2D(filters, kernel_size, padding='same')

        self.conv_add = Conv2D(filters, (1, 1), padding='same')

        self.act = Activation('relu')
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.add([x, self.conv_add(inputs)])
        x = self.act(x)

        return x

    def get_config(self):
        config = super(BottleneckBlock, self).get_config()
        config.update({
            'filters': self.conv1.filters,
            'kernel_size': self.conv1.kernel_size
        })
        return config
