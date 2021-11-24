from tensorflow import keras


class Model(keras.Model):
    def my_summary(self, input_shape):
        x = keras.layers.Input(shape=input_shape)
        return keras.Model(inputs=x, outputs=self.call(x)).summary()
