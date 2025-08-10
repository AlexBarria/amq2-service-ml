FROM apache/airflow:3.0.2

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Poetry
COPY dockerfiles/airflow/poetry.lock dockerfiles/airflow/pyproject.toml ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

# copy source code
COPY dockerfiles/airflow/secrets /opt/secrets
COPY src/airflow_dags/ /opt/airflow/dags
COPY src/domain /opt/airflow/plugins/domain
