from tensorflow import keras


class Model(keras.Model):

    def my_summary(self, input_shape):
        x = keras.layers.Input(shape=input_shape)
        return keras.Model(inputs=x, outputs=self.call(x)).summary()

    def plot_model(self, input_shape):
        x = keras.layers.Input(shape=input_shape)
        model = keras.Model(inputs=x, outputs=self.call(x))
        return keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
