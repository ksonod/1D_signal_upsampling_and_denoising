import tensorflow as tf

def residual_block(inputs, **kwargs):

    supported_kwargs = [
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "activation",
        "kernel_regularizer",
        "kernel_initializer",
        "bias_regularizer",
        "bias_initializer",
    ]
    params = {
        k: kwargs[k] for k in kwargs if k in supported_kwargs
    }
    x = tf.keras.layers.Conv1D(**params)(inputs)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(**params)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Add()([x, inputs])


def build_model(
        input_shape=(256, 1),  # (256, 1)
        num_residual_blocks=32,  # 32
        scaling_factor=4,  # 4
        filters=256,  # 256
        kernel_size=3,  # 3
        strides=1,  # 1
        padding="same",  # "same"
        kernel_regularizer=None,
        kernel_initializer=None,
        bias_regularizer=None,
        bias_initializer=None,
):
    inputs = tf.keras.Input(shape=input_shape, dtype="float32")

    # Multiple residual blocks for feature extraction
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None,
        kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
        bias_regularizer=bias_regularizer, bias_initializer=bias_initializer,
    )(inputs)
    for i in range(num_residual_blocks):
        x = residual_block(
            inputs=x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None,
            kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer, bias_initializer=bias_initializer,
        )

    # Up-sampling block
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None,
        kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
        bias_regularizer=bias_regularizer, bias_initializer=bias_initializer,
    )(x)
    x = tf.keras.layers.Conv1D(
        filters=scaling_factor, kernel_size=kernel_size, strides=strides, padding="same", activation=None,
        kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
        bias_regularizer=bias_regularizer, bias_initializer=bias_initializer
    )(x)
    x = tf.reshape(x, (-1, x.shape[1] * x.shape[2], 1))  # point shuffle.
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="FDRN")

