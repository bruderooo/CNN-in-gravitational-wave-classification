import tensorflow as tf


class ConvModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = tf.keras.layers.Conv1D(
            4096,
            3,
            activation='relu')
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_out = tf.keras.layers.Dense(1, activation='sigmoid')

    def __call__(self, x):
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dense_out(x)
        return x

    def compile(self, optimizer='adam', loss=None, metrics=None,
                loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None,
                **kwargs):
        self.compile(optimizer='adam',
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=['accuracy'])
