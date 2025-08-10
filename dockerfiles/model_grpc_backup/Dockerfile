FROM python:3.12-slim

WORKDIR /app

# Install poetry
COPY dockerfiles/model_grpc/poetry.lock dockerfiles/model_grpc/pyproject.toml .
RUN pip install --upgrade pip
RUN pip install poetry && poetry install --no-root

# Copy source code
COPY src/model_grpc/ .
COPY src/grpc_domain/ .
COPY src/domain ./domain

EXPOSE 50051
CMD ["python", "app.py"]