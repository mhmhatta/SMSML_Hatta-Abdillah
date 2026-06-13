FROM python:3.11-slim

WORKDIR /app

COPY Membangun_model/requirements.txt /app/Membangun_model/requirements.txt
RUN pip install --no-cache-dir -r /app/Membangun_model/requirements.txt

COPY Membangun_model /app/Membangun_model
COPY Monitoring_dan_Logging /app/Monitoring_dan_Logging

EXPOSE 8000

CMD ["python", "Monitoring_dan_Logging/7.Inference.py"]
