# Ejemplo de Implementación de un Servicio de creación de catálgo de moda para búsqueda.

### MLPOS2 - CEIA - FIUBA

Supongamos que trabajamos para **ML Models and something more Inc.**, la cual ofrece un servicio de creación de
catálogo de productos de moda y de búsqueda avanzada sobre el mismo tanto por texto como por imágenes utilizando el
modelo CLIP-ViT. Internamente, tanto para realizar tareas de DataOps como de MLOps, la empresa cuenta con varios
servicios que ayudan a ejecutar las acciones necesarias. También dispone de un Data Lake en S3, para este caso,
simularemos un S3 utilizando MinIO.

Para simular esta empresa, utilizaremos Docker y, a través de Docker Compose, desplegaremos varios contenedores que
representan distintos servicios en un entorno productivo.

La implementación de ese servicio incluye:

- [Apache Airflow](https://airflow.apache.org/)
    - Un DAG que obtiene datos de un repositorio público o de un repositorio local de productos de moda, realiza
      limpieza y feature engineering y guarda en un bucket s3://data los datos separados para entrenamiento y pruebas.
      MLflow hace seguimiento de este procesamiento.
    - Un DAG que realiza experimentos de fine-tuning del modelo CLIP con el dataset y se calculan métricas obtenidas. Se
      compara el nuevo modelo ajustado con el mejor modelo hasta ahora, y si es mejor, se reemplaza. Todo se lleva a
      cabo siendo registrado en MLflow.
- [MLflow](https://mlflow.org/)
- GraphQL para realizar consultas de los productos disponibles y búsquedas por texto o imágen.
- [MinIO](https://min.io/) para almacenar los buckets.
- Base de datos relacional [PostgreSQL](https://www.postgresql.org/) para almacenar los productos.
- Base de dato key-value [ValKey](https://valkey.io/)
- Aprendizaje federado y seguridad. (TBD según próxima clase)
- Orquestación del servicio en contenedores utilizando Docker.

![Diagrama de servicios](final_assign.png)

Por defecto, cuando se inician los multi-contenedores, se crean los siguientes buckets:

- `s3://data`
- `s3://mlflow` (usada por MLflow para guardar los artefactos).

y las siguientes bases de datos:

- `mlflow_db` (usada por MLflow).
- `airflow` (usada por Airflow).

## Instalación

1. Para poder levantar todos los servicios, primero instala [Docker](https://docs.docker.com/engine/install/) en tu
   computadora (o en el servidor que desees usar).
2. Clona este repositorio.
3. Crea las carpetas `airflow/config`, `airflow/dags`, `airflow/logs`, `airflow/plugins`, `airflow/logs`.
4. Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu usuario o alguno que consideres
   oportuno (para encontrar el UID, usa el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas
   internas como root y no podrás subir DAGs (en `airflow/dags`) o plugins, etc.
5. En la carpeta raíz de este repositorio, ejecuta:

```bash
docker compose --profile all up
```

6. Una vez que todos los servicios estén funcionando (verifica con el comando `docker ps -a` que todos los servicios
   estén healthy o revisa en Docker Desktop), podrás acceder a los diferentes servicios mediante:
    - Apache Airflow: http://localhost:8080
    - MLflow: http://localhost:5001
    - MinIO: http://localhost:9001 (ventana de administración de Buckets)
    - API: http://localhost:8800/
    - Documentación de la API: http://localhost:8800/docs

Si estás usando un servidor externo a tu computadora de trabajo, reemplaza `localhost` por su IP (puede ser una privada
si tu servidor está en tu LAN o una IP pública si no; revisa firewalls u otras reglas que eviten las conexiones).

Todos los puertos u otras configuraciones se pueden modificar en el archivo `.env`. Se invita a jugar y romper para
aprender; siempre puedes volver a clonar este repositorio.

## Apagar los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no se están utilizando, se
recomienda detenerlos. Para hacerlo, ejecuta el siguiente comando:

```bash
docker compose --profile all down
```

Si deseas no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), utiliza el
siguiente comando:

```bash
docker compose down --rmi all --volumes
```

Nota: Si haces esto, perderás todo en los buckets y bases de datos.

## Aspectos específicos de Airflow

### Variables de entorno

Airflow ofrece una amplia gama de opciones de configuración. En el archivo `docker-compose.yaml`, dentro de
`x-airflow-common`, se encuentran variables de entorno que pueden modificarse para ajustar la configuración de Airflow.
Pueden añadirse [otras variables](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html).

### Uso de ejecutores externos

Actualmente, para este caso, Airflow utiliza un
ejecutor [celery](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html), lo que
significa que las tareas se ejecutan en otro contenedor.

### Uso de la CLI de Airflow

Si necesitan depurar Apache Airflow, pueden utilizar la CLI de Apache Airflow de la siguiente manera:

```bash
docker compose --profile all --profile debug up
```

Una vez que el contenedor esté en funcionamiento, pueden utilizar la CLI de Airflow de la siguiente manera,
por ejemplo, para ver la configuración:

```bash
docker-compose run airflow-cli config list      
```

Para obtener más información sobre el comando, pueden
consultar [aqui](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html).

### Variables y Conexiones

Si desean agregar variables para accederlas en los DAGs, pueden hacerlo en `secrets/variables.yaml`. Para obtener
más [información](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html),
consulten la documentación.

Si desean agregar conexiones en Airflow, pueden hacerlo en `secrets/connections.yaml`. También es posible agregarlas
mediante la interfaz de usuario (UI), pero estas no persistirán si se borra todo. Por otro lado, cualquier conexión
guardada en `secrets/connections.yaml` no aparecerá en la UI, aunque eso no significa que no exista. Consulten la
documentación para obtener más
[información](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html).

## Conexión con los buckets

Dado que no estamos utilizando Amazon S3, sino una implementación local de los mismos mediante MinIO, es necesario
modificar las variables de entorno para conectar con el servicio de MinIO. Las variables de entorno son las siguientes:

```bash
AWS_ACCESS_KEY_ID=minio   
AWS_SECRET_ACCESS_KEY=minio123 
AWS_ENDPOINT_URL_S3=http://localhost:90000
```

MLflow también tiene una variable de entorno que afecta su conexión a los buckets:

```bash
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

Asegúrate de establecer estas variables de entorno antes de ejecutar tu notebook o scripts en tu máquina o en cualquier
otro lugar. Si estás utilizando un servidor externo a tu computadora de trabajo, reemplaza localhost por su dirección
IP.

Al hacer esto, podrás utilizar `boto3`, `awswrangler`, etc., en Python con estos buckets, o `awscli` en la consola.

Si tienes acceso a AWS S3, ten mucho cuidado de no reemplazar tus credenciales de AWS. Si usas las variables de entorno,
no tendrás problemas.

## Valkey

La base de datos Valkey es usada por Apache Airflow para su funcionamiento. Tal como está configurado ahora no esta
expuesto el puerto para poder ser usado externamente. Se puede modificar el archivo `docker-compose.yaml` para
habilitaro.

## Pull Request

Este repositorio está abierto para que realicen sus propios Pull Requests y así contribuir a mejorarlo. Si desean
realizar alguna modificación, **¡son bienvenidos!** También se pueden crear nuevos entornos productivos para aumentar la
variedad de implementaciones, idealmente en diferentes `branches`. Algunas ideas que se me ocurren que podrían
implementar son:

- Reemplazar Airflow y MLflow con [Metaflow](https://metaflow.org/) o [Kubeflow](https://www.kubeflow.org).
- Reemplazar MLflow con [Seldon-Core](https://github.com/SeldonIO/seldon-core).
- Agregar un servicio de tableros como, por ejemplo, [Grafana](https://grafana.com).

## Actualizaciones

Para utilizar este repositorio y cargar datos del dataset `ashraq/fashion-product-images-small`, se puede ejecutar el
siguiente comando:

```bash
poetry run python src/data/dataset_loader.py
```

Este script realiza las siguientes tareas:

- Descarga un conjunto de imágenes del dataset desde Hugging Face.
- Guarda las imágenes en un bucket S3 utilizando MinIO como almacenamiento.
- Genera un índice en PostgreSQL, preservando las columnas originales del dataset y agregando campos adicionales de
  metadatos.

---

### Verificar los datos indexados en PostgreSQL

Una vez ejecutado el script, se puede consultar el índice generado en PostgreSQL con:

```bash
psql -h localhost -p 15432 -U airflow -d airflow
```

Y luego ejecutar la siguiente consulta SQL:

```sql
SELECT * FROM fashion_files LIMIT 5;
```

---

### Instalación de `psql` en macOS

Para poder ejecutar `psql`, es necesario tenerlo instalado. En macOS, se puede instalar con:

```bash
brew install libpq
```

Como `libpq` es un paquete *keg-only*, no se agrega automáticamente al `PATH`. Para solucionarlo, se debe ejecutar:

```bash
echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verificá que haya quedado correctamente configurado con:

```bash
which psql
psql --version
```

La salida esperada debería ser similar a:

```
/opt/homebrew/opt/libpq/bin/psql
psql (PostgreSQL) 17.5
```

Con esto ya podés volver a ejecutar el comando `psql` y realizar consultas sobre la tabla `fashion_files`.

## API GraphQL

Este proyecto expone una API GraphQL desarrollada con **Strawberry** y **FastAPI**, que permite consultar los metadatos
de los archivos indexados del dataset `ashraq/fashion-product-images-small`.

### 🔌 Consultar la API

Docker levantará la API en el puerto 8801:8801. Accediendo al endpoint `/graphql` se podrán ejecutar consultas usando la
UI.

---

### 📋 Queries disponibles

#### 🔹 `allFiles`

Devuelve todos los registros indexados (limitado por defecto en el backend).

```graphql
{
  allFiles {
    id
    filename
    gender
    masterCategory
    baseColour
  }
}
```

---

#### 🔹 `filesByFilters(...)`

Consulta flexible con múltiples filtros opcionales y paginación:

**Parámetros disponibles:**

- `masterCategory` (String)
- `gender` (String)
- `baseColour` (String)
- `season` (String)
- `year` (String)
- `limit` (Int, por defecto: 50)
- `offset` (Int, por defecto: 0)

**Ejemplos:**

```graphql
{
  filesByFilters(gender: "Women", season: "Winter", limit: 10) {
    id
    filename
    productDisplayName
  }
}
```

---

### 📦 Campos disponibles en cada archivo (`FashionFile`)

- `id`
- `filename`
- `s3Path`
- `masterCategory`
- `subCategory`
- `articleType`
- `baseColour`
- `season`
- `year`
- `usage`
- `gender`
- `productDisplayName`
- `dataset`
- `created_at`

---

> 💡 Nota: los archivos físicos están almacenados en un bucket S3 (MinIO), y los campos representan metadatos extraídos
> al momento de la carga del dataset.
>> > > > > > be37c6d9145b6afd10c5928ced6e139cf350f759
