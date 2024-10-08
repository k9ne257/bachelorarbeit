#FROM python:3.10-slim
#
#WORKDIR /code
#
##COPY requirements.txt setup.pyn ./
#COPY requirements.txt ./
#
#RUN pip install --no-cache-dir -r requirements.txt
#
## Install the project itself (in editable mode, using setup.py)
##RUN pip install --no-cache-dir -e .
#
#COPY . .
#
## Set environment variables (adjust if neccessary)
#ENV PYTHONPATH="/probilix/src:${PYTHONPATH}"
#
## run tests
## CMD ["pytest"]
#
## main Python entry point, if neccessary
## CMD ["python", "./src/models/predict_model.py"]
#
#EXPOSE 8000
#
#CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-slim

WORKDIR /probilix

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/probilix/src:${PYTHONPATH}"

EXPOSE 8000

CMD ["uvicorn", "src.apis.server:app", "--host", "0.0.0.0", "--port", "8000"]
