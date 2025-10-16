FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY weather_train.py
ENTRYPOINT ["python", "weather_train.py"]