import joblib
import numpy as np
from fastapi import FastAPI

from src.Prediction_Models import features

model = joblib.load('out/iris_model.joblib')

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/money")
def read_root():
    return {"amount": "cash"}

class_names = np.array(['setosa', 'versicolor', 'virginica'])

@app.post("/predict")
def predict(data:dict):
    """

    :param data:
    """
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}
