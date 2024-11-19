import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

class uqXModel:

    def __init__(self, n_ensemble_models=3):
        """
        Initialize the uqXModel class with support for various UQ methods.
        :param n_ensemble_models: Number of models in the deep ensemble.
        """
        self.model = None
        self.ensemble_models = []  # List to store models for the ensemble
        self.n_ensemble_models = n_ensemble_models

    def train_model(self, data, model_type="cnn", uq_method="mc_dropout"):
        """
        Train a model based on the input dataset, model type, and UQ method.

        Parameters:
        - data: Tuple (X_train, y_train, X_val, y_val)
        - model_type: Type of model ("cnn", "random_forest", etc.)
        - uq_method: Type of UQ method ("bnn", "mc_dropout", "deep_ensemble", "bayes_by_backprop", etc.)
        """
        X_train, y_train, X_val, y_val = data

        if uq_method == "deep_ensemble":
            return self._apply_deep_ensemble(X_train, y_train, X_val, y_val)
        elif uq_method == "bayes_by_backprop":
            return self._apply_bayes_by_backprop(X_train, y_train, X_val, y_val)
        elif model_type == "cnn":
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

    def _apply_bayes_by_backprop(self, X_train, y_train, X_val, y_val):
        import tensorflow as tf
        import tensorflow_probability as tfp

        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))

        # Prior and posterior functions
        def prior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential([
                tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)))
            ])
            return prior_model

        def posterior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            posterior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=t[..., :n],
                    scale_diag=tf.nn.softplus(t[..., n:])))
            ])
            return posterior_model

        # Input layer
        inputs = tf.keras.Input(shape=(input_dim,))

        # Bayesian dense layer
        x = tfp.layers.DenseVariational(
            units=64,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            activation='relu')(inputs)

        # Bayesian output layer
        outputs = tfp.layers.DenseVariational(
            units=num_classes,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            activation='softmax')(x)

        # Construct model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        # Assign the trained model
        self.model = model
        return model

    def predict_with_bayes_by_backprop(self, X, num_samples=100):
        """
        Perform inference with Monte Carlo sampling for Bayes by Backprop.
        :param X: Input data.
        :param num_samples: Number of Monte Carlo samples.
        :return: Mean prediction and uncertainty (variance).
        """
        if not isinstance(self.model, tf.keras.Model):
            raise ValueError("Model is not a Bayesian neural network.")

        # Collect multiple predictions
        predictions = np.array([self.model(X, training=True).numpy() for _ in range(num_samples)])
        mean_prediction = predictions.mean(axis=0)
        variance_prediction = predictions.var(axis=0)

        return mean_prediction, variance_prediction

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
        else:
            return model

    def _apply_mc_dropout(self, model):
        """
        Apply MC Dropout for uncertainty estimation.
        """
        import tensorflow.keras.backend as K

        f_model = K.function([model.input, K.learning_phase()], [model.output])
        return f_model

    def _apply_deep_ensemble(self, X_train, y_train, X_val, y_val):
        """
        Train multiple models as part of a Deep Ensemble.
        """
        self.ensemble_models = []  # Clear any existing ensemble

        for i in range(self.n_ensemble_models):
            print(f"Training ensemble model {i+1}/{self.n_ensemble_models}")

            # Train a CNN for each ensemble member
            model = self._train_cnn(X_train, y_train, uq_method=None)
            self.ensemble_models.append(model)

        return self.ensemble_models

    def predict_with_ensemble(self, X):
        """
        Perform predictions using the Deep Ensemble.
        :param X: Input data.
        :return: Mean prediction and uncertainty (variance).
        """
        if not self.ensemble_models:
            raise ValueError("No ensemble models found. Train the ensemble first.")

        # Collect predictions from all ensemble models
        predictions = np.array([model.predict(X) for model in self.ensemble_models])
        mean_prediction = predictions.mean(axis=0)
        variance_prediction = predictions.var(axis=0)

        return mean_prediction, variance_prediction

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

    def plot_density_with_pca(self,X):
        """
        Reduce data to 1D using PCA and plot density.
        """


        # Ensure X is a numpy array or similar
        # if not isinstance(X, (np.ndarray, list)):
        #     raise TypeError("Input data must be a numerical array or list.")

        # Reduce to 1D using PCA
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X)  # Test with a subset

        # Fit Kernel Density Estimation
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(X_pca)

        # Generate values for density plot
        X_d = np.linspace(X_pca.min(), X_pca.max(), 1000).reshape(-1, 1)
        density = np.exp(kde.score_samples(X_d))

        # Plot the results
        plt.figure(figsize=(15, 6))
        plt.plot(X_d, density, label='Density')
        plt.scatter(X_pca, np.zeros_like(X_pca), alpha=0.5, label='Data Points', color='red')
        plt.title("PCA-based Density Plot for MNIST Subset")
        plt.legend()
        plt.show()

    def save_model(self, path="models/trained_model"):
        """
        Save the trained model(s) for TensorFlow MLOps deployment.
        """
        if self.ensemble_models:
            for i, model in enumerate(self.ensemble_models):
                model.save(f"{path}_ensemble_{i + 1}.h5")
        elif isinstance(self.model, tf.keras.Model):
            self.model.save(path)
        else:
            import joblib
            joblib.dump(self.model, f"{path}.pkl")