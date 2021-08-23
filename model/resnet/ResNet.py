from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

from model.resnet import IdentityBlock


class ResNet(Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, (7, 7), padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))

        self.id1a = IdentityBlock(64, (3, 3))
        self.id1b = IdentityBlock(64, (3, 3))

        self.stride_conv = Conv2D(128, (1, 1), padding='same')

        self.id2a = IdentityBlock(128, (3, 3))
        self.id2b = IdentityBlock(128, (3, 3))

        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.stride_conv(x)

        x = self.id2a(x)
        x = self.id2b(x)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x
