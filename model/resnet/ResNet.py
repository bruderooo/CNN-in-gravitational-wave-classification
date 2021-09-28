from keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

from model.resnet import IdentityBlock, StrideBlock


class ResNet(keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, (7, 7), padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))

        self.id1a = IdentityBlock(64, (3, 3))
        self.id1b = IdentityBlock(64, (3, 3))

        self.stride_conv1 = StrideBlock(128)

        self.id2a = IdentityBlock(128, (3, 3))
        self.id2b = IdentityBlock(128, (3, 3))
        #
        # self.stride_conv2 = StrideBlock(256)
        #
        # self.id3a = IdentityBlock(256, (3, 3))
        # self.id3b = IdentityBlock(256, (3, 3))
        #
        # self.stride_conv3 = StrideBlock(512)
        #
        # self.id4a = IdentityBlock(512, (3, 3))
        # self.id4b = IdentityBlock(512, (3, 3))

        self.global_pool = GlobalAveragePooling2D()

        # self.dense1 = Dense(256)
        # self.dense2 = Dense(128)
        self.dense3 = Dense(32)

        self.classifier = Dense(1, activation='sigmoid')

        self.drop = Dropout(0.2)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.stride_conv1(x)

        x = self.id2a(x)
        x = self.id2b(x)

        # x = self.stride_conv2(x)
        #
        # x = self.id3a(x)
        # x = self.id3b(x)
        #
        # x = self.stride_conv3(x)
        #
        # x = self.id4a(x)
        # x = self.id4b(x)

        x = self.global_pool(x)

        # x = self.dense1(x)
        # x = self.drop(x)
        # x = self.act(x)
        #
        # x = self.dense2(x)
        # x = self.drop(x)
        # x = self.act(x)

        x = self.dense3(x)
        x = self.drop(x)
        x = self.act(x)

        x = self.classifier(x)
        return x
