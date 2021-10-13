from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

from model.resnet import IdentityBlock


class ResNet(keras.Model):

    def __init__(self, data_format="channels_last"):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, (7, 7), padding='same', data_format=data_format)
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

    def model(self):
        x = keras.layers.Input(shape=(193, 81, 1))
        return keras.Model(inputs=[x], outputs=self.call(x))
