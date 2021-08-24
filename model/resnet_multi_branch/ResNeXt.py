from keras import Model
from keras.layers import Add
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

from model.resnet_multi_branch import IdentityBlock


class ResNeXt(Model):

    def __init__(self, filters=12, dim=256):
        super(ResNeXt, self).__init__(name='')
        self.conv = Conv2D(dim, (7, 7), padding='same')
        self.bn = BatchNormalization()
        self.max_pool = MaxPool2D((3, 3))

        self.identity_columns1a = [IdentityBlock(filters=filters, output_dim=dim) for _ in range(32)]
        self.identity_columns1b = [IdentityBlock(filters=filters, output_dim=dim) for _ in range(32)]

        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(1, activation='sigmoid')

        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)
        skip = x

        tmp = [identity_column(skip) for identity_column in self.identity_columns1a]
        tmp = self.add(tmp)
        x = self.add([tmp, skip])

        skip = x
        tmp = [identity_column(skip) for identity_column in self.identity_columns1b]
        tmp = self.add(tmp)
        x = self.add([tmp, skip])

        x = self.global_pool(x)
        x = self.classifier(x)
        return x
