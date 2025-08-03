# Ejemplo de Implementación de un Servicio de creación de catálgo de moda para búsqueda.
### MLPOS2 - CEIA - FIUBA

ML Models and something more Inc ofrece un servicio de creación de catálogo de productos de moda y de búsqueda avanzada sobre el mismo tanto por texto como por imágenes utilizando el modelo CLIP-ViT.
La implementación de ese servicio incluye:
- Apache Airflow:
  - Un DAG que obtiene datos de un repositorio público o de un repositorio local de productos de moda, realiza limpieza y feature engineering y guarda en un bucket s3://data los datos separados para entrenamiento y pruebas. MLflow hace seguimiento de este procesamiento.
  - Un DAG que realiza experimentos de fine-tuning del modelo CLIP con el dataset y se calculan métricas obtenidas. Se compara el nuevo modelo ajustado con el mejor modelo hasta ahora, y si es mejor, se reemplaza. Todo se lleva a cabo siendo registrado en MLflow.
- Un servicio REST permite crear repositorios / catálogos de productos.
- Un servicio GraphQL permite realizar llamadas de búsqueda por text<o o imágen.
- MinIO para almacenar los buckets.
- PostgreSQL para almacenar los catálogos de productos. (TBD)
- Aprendizaje federado y seguridad. (TBD según próxima clase)
- Orquestación del servicio en contenedores utilizando Docker.
