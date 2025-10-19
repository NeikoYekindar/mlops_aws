FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Make the image generic to run any project script.
# Jenkins will invoke specific scripts like weather_train_1.py, weather_train_2.py, or weather_test.py.
ENTRYPOINT ["python"]
