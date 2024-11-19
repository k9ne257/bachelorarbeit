import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class uqXBnnModel:
    def __init__(self, input_dim, hidden_size=32, kl_weight=1.0):
        """
        Initialize the Bayesian Neural Network model.

        Parameters:
        - input_dim: Integer, dimensionality of the input features.
        - hidden_size: Integer, number of units in the hidden layers.
        - kl_weight: Float, weight for the KL divergence loss term.
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.model = self._build_model()

    def _prior_fn(self, kernel_size, bias_size=0, dtype=None):
        """
        Define the prior distribution function for the weights.

        Parameters:
        - kernel_size: Integer, size of the kernel weights.
        - bias_size: Integer, size of the bias weights.
        - dtype: Data type of the weights.

        Returns:
        - prior: tfd.Distribution instance representing the prior.
        """
        n = kernel_size + bias_size
        return tfp.distributions.Independent(
            tfp.distributions.Normal(
                loc=tf.zeros(n, dtype=dtype),
                scale=1.0
            ), reinterpreted_batch_ndims=1
        )

    def _posterior_fn(self, kernel_size, bias_size=0, dtype=None):
        """
        Define the posterior (variational) distribution function for the weights.

        Parameters:
        - kernel_size: Integer, size of the kernel weights.
        - bias_size: Integer, size of the bias weights.
        - dtype: Data type of the weights.

        Returns:
        - posterior: A Keras Sequential model representing the posterior distribution.
        """
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.0))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ), reinterpreted_batch_ndims=1
                )
            )
        ])

    def _build_model(self):
        """
        Build the Bayesian neural network model using TensorFlow Probability.

        Returns:
        - model: A compiled Keras model.
        """
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = tfp.layers.DenseVariational(
            units=self.hidden_size,
            make_prior_fn=self._prior_fn,
            make_posterior_fn=self._posterior_fn,
            kl_weight=self.kl_weight,
            activation='relu'
        )(inputs)

        x = tfp.layers.DenseVariational(
            units=self.hidden_size,
            make_prior_fn=self._prior_fn,
            make_posterior_fn=self._posterior_fn,
            kl_weight=self.kl_weight,
            activation='relu'
        )(x)

        outputs = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=self._prior_fn,
            make_posterior_fn=self._posterior_fn,
            kl_weight=self.kl_weight,
            activation=None
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, learning_rate=0.01):
        """
        Train the Bayesian neural network model.

        Parameters:
        - X_train: Training input data.
        - y_train: Training target data.
        - X_val: Validation input data (optional).
        - y_val: Validation target data (optional).
        - epochs: Number of training epochs.
        - batch_size: Size of each training batch.
        - learning_rate: Learning rate for the optimizer.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._neg_log_likelihood_loss
        )

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def _neg_log_likelihood_loss(self, y_true, y_pred):
        """
        Negative log-likelihood loss function for regression tasks.

        Parameters:
        - y_true: True target values.
        - y_pred: Predicted target values.

        Returns:
        - Loss value.
        """
        # For regression, we can use Mean Squared Error as the NLL
        return tf.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred))

    def predict(self, X_test, num_samples=100):
        """
        Make predictions using the trained Bayesian neural network with uncertainty estimation.

        Parameters:
        - X_test: Input data for prediction.
        - num_samples: Number of Monte Carlo samples for uncertainty estimation.

        Returns:
        - mean_prediction: Mean predictions over the samples.
        - std_prediction: Standard deviation of the predictions (uncertainty).
        """
        predictions = np.array([self.model(X_test, training=True).numpy() for _ in range(num_samples)])
        mean_prediction = predictions.mean(axis=0)
        std_prediction = predictions.std(axis=0)
        return mean_prediction, std_prediction

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        - filepath: String, path to save the model.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Load a trained model from a file.

        Parameters:
        - filepath: String, path to the saved model.
        """
        self.model = tf.keras.models.load_model(filepath, custom_objects={'DenseVariational': tfp.layers.DenseVariational})