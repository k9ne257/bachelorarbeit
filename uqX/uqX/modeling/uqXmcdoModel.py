import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class uqXmcdoModel(nn.Module):
    def __init__(self, input_dim=1, hidden_size=32, dropout_rate=0.1):
        """
        Initialize the uqXmcdoModel class.

        Parameters:
        - input_dim: Number of input features.
        - hidden_size: Number of units in hidden layers.
        - dropout_rate: Dropout rate for uncertainty estimation.
        """
        super(uqXmcdoModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass with dropout applied for uncertainty estimation.
        """
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.out(x)

    def train_model(self, train_loader, epochs=150, learning_rate=0.01):
        """
        Train the model using a dataset loader.

        Parameters:
        - train_loader: DataLoader for training data.
        - epochs: Number of training epochs.
        - learning_rate: Learning rate for the optimizer.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Loss: {loss.item():.4f}")

    def make_mcdo_plot(self, x_test, y_test=None, mc_samples=50, start=None, end=None):
        """
        Generate a plot showing the model's predictions with uncertainty.

        Parameters:
        - x_test: Input test data as a PyTorch tensor or numpy array.
        - y_test: Optional, true target values as a PyTorch tensor or numpy array.
        - mc_samples: Number of Monte Carlo samples for uncertainty estimation.
        - start, end: Optional, x-values to mark the training range.
        """
        # Ensure x_test is a torch.Tensor
        if isinstance(x_test, np.ndarray):
            x_test = torch.from_numpy(x_test).float()

        # Ensure x_test has the correct shape
        if x_test.dim() == 1:
            x_test = x_test.view(-1, 1)

        self.eval()
        predictions = []

        # Generate multiple predictions with MC Dropout
        self.train()  # Enable dropout during prediction
        predictions = []

        with torch.no_grad():
            for _ in range(mc_samples):
                preds = self(x_test)
                predictions.append(preds)

        self.eval()  # Return to eval mode after sampling

        preds_tensor = torch.stack(predictions)  # Shape: (mc_samples, batch_size, 1)
        mean_preds = preds_tensor.mean(dim=0).squeeze().numpy()
        std_preds = preds_tensor.std(dim=0).squeeze().numpy()

        # Prepare data for plotting
        x_values = x_test.squeeze().numpy()
        y_intervals = [mean_preds, mean_preds + 2 * std_preds, mean_preds - 2 * std_preds]
        dataframes = []
        labels = ["Mean Prediction", "Mean + 2 Std", "Mean - 2 Std"]

        for i, y_vals in enumerate(y_intervals):
            df = pd.DataFrame({"x": x_values, "y": y_vals, "label": labels[i]})
            dataframes.append(df)

        plot_data = pd.concat(dataframes).reset_index(drop=True)

        # Automatically set start and end if not provided
        if start is None:
            start = x_values.min()
        if end is None:
            end = x_values.max()

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=plot_data, x="x", y="y", hue="label", style="label")

        # Highlight training range if specified
        plt.axvline(x=start, color='red', linestyle='--', label='Train Start')
        plt.axvline(x=end, color='green', linestyle='--', label='Train End')

        # Overlay test data
        if y_test is not None:
            if isinstance(y_test, torch.Tensor):
                y_test = y_test.numpy()
            plt.scatter(x_values, y_test, color="blue", marker="o", alpha=0.5, label="Test Data")

        plt.title("Monte Carlo Dropout Predictions with Uncertainty")
        plt.xlabel("Input")
        plt.ylabel("Prediction")
        plt.legend()
        plt.show()

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        - filepath: Path to save the model state.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a saved model state from a file.

        Parameters:
        - filepath: Path to the saved model state.
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()
        print(f"Model loaded from {filepath}")
