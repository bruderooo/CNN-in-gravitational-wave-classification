from keras import Model
from keras.layers import Add, Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

from model.resnet_multi_branch import BlocksGroup


class ResNeXt(Model):

    def __init__(self, filters=3, dim=64):
        super(ResNeXt, self).__init__(name='')
        self.conv = Conv2D(dim, (2 * filters + 1, 2 * filters + 1), padding='same')
        self.bn = BatchNormalization()
        self.max_pool = MaxPool2D((3, 3))

        self.identity_columns1a = BlocksGroup(filters=filters, dim=dim)
        self.identity_columns1b = BlocksGroup(filters=filters, dim=dim)

        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(1, activation='sigmoid')

        self.act = Activation('relu')
        self.add = Add()

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
