# dockerfiles/graphql_api/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install poetry
COPY dockerfiles/graphql_api/poetry.lock dockerfiles/graphql_api/pyproject.toml ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

# Copy source code
COPY src/graphql_api/ .
COPY src/grpc_domain/ .

EXPOSE 8801

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8801"]