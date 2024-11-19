import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import stats

sns.set(rc={'figure.figsize': (9, 7)})


class uqXAleatoricModel(tf.keras.Model):
    def __init__(self):
        super(uqXAleatoricModel, self).__init__()
        hidden_size = 64

        # Define layers
        self.fc1 = tf.keras.layers.Dense(hidden_size)
        self.fc2 = tf.keras.layers.Dense(hidden_size)
        self.mu = tf.keras.layers.Dense(1)
        self.var = tf.keras.layers.Dense(1)

        # Placeholder for training data (optional)
        self.x_train = None
        self.y_train = None

    def call(self, x):
        h = tf.math.tanh(self.fc1(x))
        h = tf.math.tanh(self.fc2(h))
        mu = self.mu(h)
        var = tf.math.softplus(self.var(h))  # Ensure variance is positive
        return mu, var

    def nll_loss(self, y_true, y_pred_mu, y_pred_var):
        loss = 0.5 * tf.math.log(y_pred_var) + \
               0.5 * ((y_true - y_pred_mu) ** 2) / y_pred_var
        return tf.reduce_mean(loss)

    def train_model(self, x_train, y_train, epochs=1000, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Build the model (if not already built)
        if not self.built:
            self.build(input_shape=x_train.shape)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                mu, var = self(x_train)
                loss = self.nll_loss(y_train, mu, var)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    def make_aleatoric_plot(self, x_test, y_test, start=None, end=None):
        # Get predictions
        mu, var = self(x_test)
        mu, sigma = mu.numpy(), var.numpy() ** 0.5

        # ~95% confidence interval
        y_vals = [mu.squeeze(), (mu + 2 * sigma).squeeze(), (mu - 2 * sigma).squeeze()]
        dfs = []

        # Create DataFrame from predictions
        for i in range(3):
            data = {
                "x": x_test.numpy().squeeze(),
                "y": y_vals[i]
            }
            temp = pd.DataFrame(data)
            dfs.append(temp)
        df = pd.concat(dfs).reset_index(drop=True)

        # Plot predictions with confidence intervals
        plt.figure(figsize=(9, 7))
        sns.lineplot(data=df, x="x", y="y")

        # Highlight training range (if provided)
        if start is not None:
            plt.axvline(x=start, color='red', linestyle='--', label='Training Range Start')
        if end is not None:
            plt.axvline(x=end, color='red', linestyle='--', label='Training Range End')

        # Plot test data and training data
        plt.scatter(x_test.numpy(), y_test, c="green", marker="*", alpha=0.3, label="Test Data")
        if self.x_train is not None and self.y_train is not None:
            plt.scatter(self.x_train.numpy(), self.y_train.numpy(), c="blue", marker="o", alpha=0.5, label="Train Data")
        plt.title("Model Predictions with Aleatoric Uncertainty")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()