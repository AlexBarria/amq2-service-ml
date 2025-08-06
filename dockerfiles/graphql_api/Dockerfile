# dockerfiles/graphql_api/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install poetry
COPY poetry.lock pyproject.toml ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

# Copy source code
COPY src ./src

EXPOSE 8801

CMD ["uvicorn", "src.api.graphql_api:app", "--host", "0.0.0.0", "--port", "8801"]