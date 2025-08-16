FROM python:3.12-slim

WORKDIR /app

# Install poetry
COPY dockerfiles/rest_api/poetry.lock dockerfiles/rest_api/pyproject.toml ./
RUN apt-get update && apt-get install -y curl
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

# Copy source code
COPY src/rest_api/ .
COPY src/grpc_domain/ .

EXPOSE 8800