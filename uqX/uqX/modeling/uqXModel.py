import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from uqX.modeling.BayesianConverter import BayesianConverter


class uqXModel:

    def __init__(self):
        self.model = None
        self.uq_method = None


    def train_model(self, data, model_type="cnn", uq_method="mc_dropout"):
        """
        Train a model based on the input dataset, model type, and UQ method.

        Parameters:
        - data: Tuple (X_train, y_train, X_val, y_val)
        - model_type: Type of model ("cnn", "random_forest", etc.)
        - uq_method: Type of UQ method ("bnn", "mc_dropout", etc.)
        """
        X_train, y_train, X_val, y_val = data

        if model_type == "cnn":
            self.model = self._train_cnn(X_train, y_train, uq_method)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier().fit(X_train, y_train)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression().fit(X_train, y_train)
        elif model_type == "linear_regression":
            self.model = LinearRegression().fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Evaluate model (optional)
        print("Validation Score:", self.model.score(X_val, y_val))

        return self.model


    def _train_cnn(self, X_train, y_train, uq_method):
        """
        Train a CNN model with optional UQ methods.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        if uq_method == "mc_dropout":
            return self._apply_mc_dropout(model)
        elif uq_method == "bnn":
            return self._apply_bnn(model)
        else:
            return model


    def _apply_mc_dropout(self, model):
        """
        Apply MC Dropout for uncertainty estimation.
        """
        import tensorflow.keras.backend as K

        f_model = K.function([model.input, K.learning_phase()], [model.output])
        return f_model

    def _apply_bnn(self, X_train, y_train, X_val, y_val):
        """
        Train a Bayesian Neural Network (BNN) using TensorFlow Probability.
        """
         # Define the number of features and output classes

        input_dim = X_train.shape[1]
        num_classes = len(tf.unique(y_train)[0])

        # Define Bayesian layers
        def make_prior_fn():
            """Define a normal prior over the weights."""
            return tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([input_dim, num_classes]),
                                                            scale_diag=tf.ones([input_dim, num_classes]))

        def make_posterior_fn():
            """Define a posterior with trainable parameters."""
            return tfp.layers.MultivariateNormalTriL(num_components=1)

        # Define the Bayesian Neural Network
        bnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tfp.layers.DenseVariational(
                units=64,
                make_prior_fn=make_prior_fn,
                make_posterior_fn=make_posterior_fn,
                activation='relu'
            ),
            tfp.layers.DenseVariational(
                units=num_classes,
                make_prior_fn=make_prior_fn,
                make_posterior_fn=make_posterior_fn,
                activation='softmax'
            )
        ])

        # Compile the model
        bnn.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Train the model
        bnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

        # Assign the trained model
        self.model = bnn

        return bnn



    def plot_density(X, title="Data Density", bandwidth=1.0):
        """
        Plot the density of data points using Kernel Density Estimation (KDE).
        """
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        X_d = np.linspace(X.min(), X.max(), 1000)[:, np.newaxis]
        density = np.exp(kde.score_samples(X_d))

        plt.figure(figsize=(8, 6))
        plt.plot(X_d, density, label='Density')
        plt.scatter(X, np.zeros_like(X), alpha=0.5, label='Data Points', color='red')
        plt.title(title)
        plt.legend()
        plt.show()

    def save_model(self, path="models/trained_model"):
        """
        Save the trained model for TensorFlow MLOps deployment.
        """
        if isinstance(self.model, tf.keras.Model):
            self.model.save(path)
        else:
            import joblib
            joblib.dump(self.model, f"{path}.pkl")
