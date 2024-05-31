import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1):
        super().__init__()

        if (alpha < 0) or (1 < alpha):
            raise ValueError("alpha should be between 0 and 1.")
        self.alpha = alpha

    def get_config(self):
        return {
            "name": "custom_loss",
            "alpha": self.alpha
        }

    def __call__(self, y_true, y_pred, sample_weight=None):

        loss1 = tf.keras.losses.MeanSquaredError()(y_true, tf.reshape(y_pred, shape=y_true.shape))

        loss2 = tf.keras.losses.MeanAbsoluteError()(
            tf.experimental.numpy.diff(y_true, axis=1),
            tf.experimental.numpy.diff(tf.reshape(y_pred, shape=y_true.shape), axis=1)
        )

        return self.alpha * loss1 + (1 - self.alpha) * loss2
