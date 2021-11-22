from tensorflow import keras


class PaperModel(keras.Model):

    def __init__(self):
        super(PaperModel, self).__init__()

        self.time_axis_conv1 = keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding='same')
        self.time_bn1 = keras.layers.BatchNormalization()
        self.time_axis_conv2 = keras.layers.Conv2D(filters=64, kernel_size=(1, 3), padding='same')
        self.time_bn2 = keras.layers.BatchNormalization()

        self.freq_axis_conv1 = keras.layers.Conv2D(filters=32, kernel_size=(5, 1), padding='same')
        self.freq_bn1 = keras.layers.BatchNormalization()
        self.freq_axis_conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 1), padding='same')
        self.freq_bn2 = keras.layers.BatchNormalization()

        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3))
        self.bn2 = keras.layers.BatchNormalization()
        self.max_pool = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3))
        self.bn3 = keras.layers.BatchNormalization()

        self.global_avg_pool = keras.layers.GlobalAveragePooling2D()
        self.dense1 = keras.layers.Dense(units=128)
        self.dense2 = keras.layers.Dense(units=1, activation='sigmoid')

        self.leaky_relu = keras.layers.LeakyReLU(0.1)

    def call(self, inputs):
        x_time = self.time_axis_conv1(inputs)
        x_time = self.time_bn1(x_time)
        x_time = self.leaky_relu(x_time)
        x_time = self.time_axis_conv2(x_time)
        x_time = self.time_bn2(x_time)
        x_time = self.leaky_relu(x_time)

        x_freq = self.freq_axis_conv1(inputs)
        x_freq = self.freq_bn1(x_freq)
        x_freq = self.leaky_relu(x_freq)
        x_freq = self.freq_axis_conv2(x_freq)
        x_freq = self.freq_bn2(x_freq)
        x_freq = self.leaky_relu(x_freq)

        x = keras.layers.concatenate([x_time, x_freq])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
