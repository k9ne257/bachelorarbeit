FROM python:3.10-slim

WORKDIR /code

#COPY requirements.txt setup.pyn ./
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Install the project itself (in editable mode, using setup.py)
RUN pip install --no-cache-dir -e .

# Copy the rest of the project into the container
COPY . .

# Set environment variables (adjust if neccessary)
ENV PYTHONPATH="/code/src:${PYTHONPATH}"

# run tests
# CMD ["pytest"]

# main Python entry point, if neccessary
# CMD ["python", "./src/models/predict_model.py"]

EXPOSE 8080
