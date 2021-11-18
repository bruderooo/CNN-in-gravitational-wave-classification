from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Dropout, MaxPool2D


class SmallConvModel(keras.Model):

    def __init__(self):
        super(SmallConvModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool = MaxPool2D(pool_size=(2, 2))
        self.flatten = GlobalAveragePooling2D()

        self.d1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.d2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout(x)
        return self.d2(x)
