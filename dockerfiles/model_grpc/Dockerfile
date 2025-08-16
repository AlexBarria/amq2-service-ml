FROM python:3.12-slim

WORKDIR /app

# Install requirements
COPY dockerfiles/model_grpc/requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

# Copy source code
COPY src/model_grpc/ .
COPY src/grpc_domain/ .
COPY src/domain ./domain

EXPOSE 50051
CMD ["python", "app.py"]