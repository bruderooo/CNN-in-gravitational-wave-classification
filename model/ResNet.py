from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense, \
    Add, Dropout


class ResNet(keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, (7, 7), padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))

        self.id1a = IdentityBlock(64, (3, 3))
        self.id1b = IdentityBlock(64, (3, 3))

        self.bottleneck1 = BottleneckBlock(128, (3, 3))
        self.id2a = IdentityBlock(128, (3, 3))
        self.id2b = IdentityBlock(128, (3, 3))

        self.bottleneck2 = BottleneckBlock(256, (3, 3))
        self.id3a = IdentityBlock(256, (3, 3))
        self.id3b = IdentityBlock(256, (3, 3))

        self.global_pool = GlobalAveragePooling2D()
        self.drop = Dropout(0.5)

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

        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
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
        return {}


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


class BottleneckBlock(keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super(BottleneckBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.conv_add = Conv2D(filters, (1, 1), padding='same')

        self.act = Activation('relu')
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, self.conv_add(inputs)])
        x = self.act(x)

        return x

    def get_config(self):
        return {
            'filters': self.conv1.filters,
            'kernel_size': self.conv1.kernel_size
        }
