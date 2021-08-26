from tensorflow import keras
from tensorflow.keras import layers

from model.resnet_multi_branch import IdentityBlock


class BlocksGroup(keras.Model):

    def __init__(self, filters, dim):
        super(BlocksGroup, self).__init__()

        self.identity_columns = [IdentityBlock(filters=filters, output_dim=dim) for _ in range(16)]
        self.add = layers.Add()

    def call(self, inputs):
        x = [identity_column(inputs) for identity_column in self.identity_columns1a]
        x = self.add(x)
        x = self.add([x, inputs])

        return x
