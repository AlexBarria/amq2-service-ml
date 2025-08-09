FROM python:3.12-slim

WORKDIR /app
COPY app/ .
COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 50051
CMD ["python", "main.py"]