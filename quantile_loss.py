import tensorflow as tf


class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, q, name="quantile_loss"):
        super(QuantileLoss, self).__init__(name=name)
        if q < 0 or q > 1:
            raise ValueError("Quantile value q must be between 0 and 1.")
        self.q = q

    def call(self, y_true, y_pred):
        # Calculate the difference between true and predicted values
        error = y_true - y_pred

        # Elements where prediction is greater than the true value
        loss_positive = self.q * error

        # Elements where prediction is less than the true value
        loss_negative = (1 - self.q) * -error

        # Combine positive and negative losses
        return tf.reduce_mean(tf.maximum(loss_positive, loss_negative))

    def get_config(self):
        # Obtain the base configuration
        config = super(QuantileLoss, self).get_config()
        # Add the 'q' quantile parameter to the configuration
        config["q"] = self.q
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
