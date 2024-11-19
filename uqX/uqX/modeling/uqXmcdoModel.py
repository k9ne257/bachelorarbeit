import torch
import torch.nn as nn
import torch.optim as optim
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@variational_estimator
class uqXBnnModel(nn.Module):
    def __init__(self, input_dim, hidden_size=32):
        """
        Initialize the Bayesian Neural Network model.

        Parameters:
        - input_dim: Integer, dimensionality of the input features.
        - hidden_size: Integer, number of units in the hidden layers.
        """
        super(uqXBnnModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Define the Bayesian layers
        self.blinear1 = BayesianLinear(self.input_dim, self.hidden_size)
        self.blinear2 = BayesianLinear(self.hidden_size, self.hidden_size)
        self.blinear3 = BayesianLinear(self.hidden_size, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.blinear1(x))
        x = self.relu(self.blinear2(x))
        x = self.blinear3(x)
        return x

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, learning_rate=0.01):
        """
        Train the Bayesian neural network model.

        Parameters:
        - X_train: Training input data as a NumPy array or PyTorch tensor.
        - y_train: Training target data as a NumPy array or PyTorch tensor.
        - X_val: Validation input data (optional).
        - y_val: Validation target data (optional).
        - epochs: Number of training epochs.
        - batch_size: Size of each training batch.
        - learning_rate: Learning rate for the optimizer.
        """
        # Convert data to PyTorch tensors if they aren't already
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if X_val is not None and isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if y_val is not None and isinstance(y_val, np.ndarray):
            y_val = torch.from_numpy(y_val).float()

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                loss = self.sample_elbo(
                    inputs=batch_X,
                    labels=batch_y,
                    criterion=criterion,
                    sample_nbr=3,
                    complexity_cost_weight=1 / len(X_train)
                )
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    def predict(self, X_test, num_samples=100):
        """
        Make predictions using the trained Bayesian neural network with uncertainty estimation.

        Parameters:
        - X_test: Input data for prediction as a NumPy array or PyTorch tensor.
        - num_samples: Number of Monte Carlo samples for uncertainty estimation.

        Returns:
        - mean_prediction: Mean predictions over the samples.
        - std_prediction: Standard deviation of the predictions (uncertainty).
        """
        # Convert data to PyTorch tensor if it isn't already
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()

        self.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds = self(X_test)
                predictions.append(preds.numpy())
        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        std_prediction = predictions.std(axis=0)
        return mean_prediction, std_prediction

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        - filepath: String, path to save the model.
        """
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """
        Load a trained model from a file.

        Parameters:
        - filepath: String, path to the saved model.
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()

    def make_bnn_plot(self, X_test, y_test=None, samples=500, start=None, end=None):
        """
        Generate a plot showing the Bayesian neural network's predictions with uncertainty.

        Parameters:
        - X_test: Input data for prediction as a NumPy array or PyTorch tensor.
        - y_test: True target values (optional) as a NumPy array or PyTorch tensor.
        - samples: Number of Monte Carlo samples for uncertainty estimation.
        - start: Optional, x-value to mark the start of a highlighted region.
        - end: Optional, x-value to mark the end of a highlighted region.
        """
        if isinstance(X_test, np.ndarray):
            tensor_x_test = torch.from_numpy(X_test).float()
        else:
            tensor_x_test = X_test.float()

            # Ensure tensor_x_test has shape (num_samples, input_dim)
        if tensor_x_test.dim() == 1:
            tensor_x_test = tensor_x_test.unsqueeze(1)

        # Generate predictions
        self.eval()
        preds = []
        with torch.no_grad():
            for _ in range(samples):
                pred = self(tensor_x_test)
                preds.append(pred.squeeze())
        preds = torch.stack(preds)

        # Calculate mean and standard deviation
        means = preds.mean(axis=0).detach().numpy()
        stds = preds.std(axis=0).detach().numpy()

        # Prepare data for plotting
        x_values = tensor_x_test.squeeze().numpy()
        y_vals = [means, means + 2 * stds, means - 2 * stds]
        dfs = []
        labels = ['Mean Prediction', 'Mean + 2 Std', 'Mean - 2 Std']

        for i in range(3):
            data = {
                "x": x_values,
                "y": y_vals[i],
                "label": labels[i]
            }
            temp = pd.DataFrame(data)
            dfs.append(temp)

        df = pd.concat(dfs)

        # Plot predictions with confidence intervals
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="x", y="y", hue="label", style="label")

        # Highlight training range if specified
        if start is not None:
            plt.axvline(x=start, color='red', linestyle='--', label='Start')
        if end is not None:
            plt.axvline(x=end, color='green', linestyle='--', label='End')

        # Plot test data points if provided
        if y_test is not None:
            if isinstance(y_test, torch.Tensor):
                y_test = y_test.numpy()
            plt.scatter(X_test, y_test, c="green", marker="*", alpha=0.5, label='Test Data')

        plt.title("Bayesian Neural Network Predictions with Uncertainty")
        plt.xlabel("Input")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()