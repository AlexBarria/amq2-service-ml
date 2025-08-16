FROM python:3.12-slim


COPY poetry.lock pyproject.toml ./

RUN apt-get update && apt-get install -y wget
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

EXPOSE 5000