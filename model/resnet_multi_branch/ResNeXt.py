from keras import Model
from keras.layers import Add
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

from model.resnet_multi_branch import IdentityBlock


class ResNeXt(Model):

    def __init__(self, filters=4, input_size=256):
        super(ResNeXt, self).__init__(name='')
        self.conv = Conv2D(64, (7, 7), padding='same')
        self.bn = BatchNormalization()

        self.identity_columns1a = [IdentityBlock(filters, input_size) for _ in range(32)]
        self.identity_columns1b = [IdentityBlock(filters, input_size) for _ in range(32)]

        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(1, activation='sigmoid')

        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        tmp = [identity_column(x) for identity_column in self.identity_columns]
        x = self.add(tmp)
        x = self.add([x, input_tensor])

        tmp = [identity_column(x) for identity_column in self.identity_columns]
        x = self.add(tmp)
        x = self.add([x, input_tensor])

        x = self.global_pool(x)
        x = self.classifier(x)
        return x
