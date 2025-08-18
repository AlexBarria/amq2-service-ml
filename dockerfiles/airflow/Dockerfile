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
COPY dockerfiles/airflow/requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt

# copy source code
COPY dockerfiles/airflow/secrets /opt/secrets
