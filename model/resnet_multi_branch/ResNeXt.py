from tensorflow import keras
from tensorflow.keras import layers

from model.resnet_multi_branch import BlocksGroup


class ResNeXt(keras.Model):

    def __init__(self, filters=3, dim=64):
        super(ResNeXt, self).__init__()
        self.conv = layers.Conv2D(dim, (2 * filters + 1, 2 * filters + 1), padding='same')
        self.bn = layers.BatchNormalization()
        self.max_pool = layers.MaxPool2D((3, 3))

        self.identity_columns1a = BlocksGroup(filters=filters, dim=dim)
        self.identity_columns1b = BlocksGroup(filters=filters, dim=dim)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(1, activation='sigmoid')

        self.act = layers.Activation('relu')
        self.add = layers.Add()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.identity_columns1a(x)
        x = self.identity_columns1b(x)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x
