import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class BayesianConverter:
    def __init__(self):
        pass

    def convert_to_bnn(self, model, prior_stddev=1.0, posterior_stddev=0.1):
        """
        Converts a given model into a Bayesian Neural Network (BNN).

        Parameters:
        - model: The input TensorFlow/Keras model to be converted.
        - prior_stddev: Standard deviation for the prior distribution.
        - posterior_stddev: Initial standard deviation for the posterior distribution.

        Returns:
        - A new model with Bayesian layers.
        """
        def make_prior_fn(stddev):
            """Defines the prior distribution for Bayesian layers."""
            return tfd.Normal(loc=0.0, scale=stddev)

        def make_posterior_fn(stddev):
            """Defines the posterior distribution for Bayesian layers."""
            return tfp.layers.MultivariateNormalTriL(num_components=1, event_shape=[])

        # Iterate over the layers and replace them
        bnn_layers = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Replace Dense layers with DenseVariational
                bnn_layers.append(
                    tfp.layers.DenseVariational(
                        units=layer.units,
                        make_prior_fn=lambda: make_prior_fn(prior_stddev),
                        make_posterior_fn=lambda: make_posterior_fn(posterior_stddev),
                        activation=layer.activation
                    )
                )
            elif isinstance(layer, tf.keras.layers.Conv2D):
                # Replace Conv2D layers with Conv2DVariational
                bnn_layers.append(
                    tfp.layers.Convolution2DVariational(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        make_prior_fn=lambda: make_prior_fn(prior_stddev),
                        make_posterior_fn=lambda: make_posterior_fn(posterior_stddev),
                        activation=layer.activation,
                        strides=layer.strides,
                        padding=layer.padding
                    )
                )
            else:
                # Keep non-trainable layers as they are (e.g., Input, Dropout)
                bnn_layers.append(layer)

        # Build a new model
        bnn_model = tf.keras.Sequential(bnn_layers)
        return bnn_model
