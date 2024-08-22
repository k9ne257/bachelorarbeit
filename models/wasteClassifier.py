# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KernelDensity

# Fetch the dataset
from ucimlrepo import fetch_ucirepo
realwaste = fetch_ucirepo(id=908)
X = realwaste.data.features
y = realwaste.data.targets

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Evaluate the models
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# Quantify prediction certainty using Kernel Density Estimation
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(X_train_scaled)
log_density = kde.score_samples(X_test_scaled)
certainty = np.exp(log_density)

# Add certainty to the prediction results
results = pd.DataFrame({
    'True Label': y_test,
    'LR Prediction': y_pred_lr,
    'RF Prediction': y_pred_rf,
    'Certainty': certainty
})

print(results.head())

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(certainty)
plt.title('Density Plot of Prediction Certainty')
plt.xlabel('Certainty')
plt.ylabel('Density')
plt.show()