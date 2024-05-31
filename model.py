import tensorflow as tf

def residual_block(inputs, **kwargs):

    supported_kwargs = ["filters", "kernel_size", "strides", "padding", "activation"]
    params = {
        k: kwargs[k] for k in kwargs if k in supported_kwargs
    }
    params["kernel_initializer"] = tf.keras.initializers.HeNormal(
        tf.random.uniform(shape=[], maxval=100, dtype=tf.int32, seed=10)
    )
    x = tf.keras.layers.Conv1D(**params)(inputs)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(**params)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Add()([x, inputs])


def build_model(input_shape=(256, 1), num_residual_blocks=32, filters=256, scaling_factor=4, kernel_size=3, strides=1):
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")

    # Multiple residual blocks for feature extraction
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None)(inputs)
    for i in range(num_residual_blocks):
        x = residual_block(inputs=x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None)

    # Up-sampling block
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None)(x)
    x = tf.keras.layers.Conv1D(filters=scaling_factor, kernel_size=kernel_size, strides=strides, padding="same", activation=None)(x)
    x = tf.reshape(x, (-1, x.shape[1] * x.shape[2], 1))  # point shuffle.
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="FDRN")

