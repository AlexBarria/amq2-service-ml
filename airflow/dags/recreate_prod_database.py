import datetime
from airflow.decorators import dag, task

markdown_text = """
### ETL Process to recreate the Fashion Data prod database.

This DAG download and prepare the Fashion Product Images dataset from Hugging Face.

After preprocessing, it saves all metadata and corresponding images into an S3 bucket.
"""

default_args = {
    'owner': "Alex Barria, Clara Bureu, Maximiliano Torti",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="recreate_prod_database",
    description="ETL Process to recreate the Fashion Data prod database.",
    doc_md=markdown_text,
    tags=["ETL", "Fashion", "Dataset"],
    default_args=default_args,
    catchup=False,
)
def recreate_prod_database():
    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw data from Hugging Face.
        """
        from datasets import load_dataset
        from airflow.models import Variable
        import logging

        logger = logging.getLogger(__name__)
        logger.info('Downloading dataset from Hugging Face')

        # Load dataset
        try:
            dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
        except Exception as e:
            logger.error(f'Error loading dataset: {e}')
            raise

        logger.info('Dataset downloaded')

        # Save information of the dataset
        storage_options = {"client_kwargs": {"endpoint_url": "http://s3:9000"}}
        dataset.save_to_disk("s3://prod/raw/fashion-product", storage_options=storage_options)
        logger.info('Dataset saved in s3')

    @task.virtualenv(
        task_id="process_datasets",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def process_dataset():
        """
        Process the dataset by creating the products metadata and saving images.
        """
        from datasets import load_from_disk
        from airflow.models import Variable
        import boto3
        import logging
        import io
        import datetime
        from sqlalchemy import create_engine, Column, Integer, String, DateTime, MetaData, Table
        from pgvector.sqlalchemy import Vector

        logger = logging.getLogger(__name__)

        storage_options = {"client_kwargs": {"endpoint_url": "http://s3:9000"}}
        s3_client = boto3.client('s3')

        engine = create_engine(Variable.get("fashion_db_conn"))
        metadata = MetaData()
        table_products = Table("fashion_files", metadata,
                               Column("id", Integer, primary_key=True, autoincrement=True),
                               Column("filename", String),
                               Column("s3_path", String),
                               Column("masterCategory", String),
                               Column("subCategory", String),
                               Column("articleType", String),
                               Column("baseColour", String),
                               Column("season", String),
                               Column("year", String),
                               Column("usage", String),
                               Column("gender", String),
                               Column("productDisplayName", String),
                               Column("dataset", String),
                               Column("created_at", DateTime),
                               Column("embedding", Vector(512), nullable=True))
        metadata.drop_all(engine, checkfirst=True)
        metadata.create_all(engine)

        # Process dataset
        logger.info("Processing dataset...")
        dataset = load_from_disk("s3://prod/raw/fashion-product", storage_options=storage_options)
        s3_dir = "fashion_files/"
        for idx, item in enumerate(dataset):
            item = dict(item)  # Force conversion to dict for compatibility
            item_name = f"{idx}_{item['id']}.jpg"
            image_s3_path = s3_dir + item_name
            image_bytes = io.BytesIO()
            item['image'].save(image_bytes, 'JPEG')
            image_bytes.seek(0)
            s3_client.upload_fileobj(image_bytes, "prod", image_s3_path)
            with engine.begin() as conn:
                conn.execute(table_products.insert().values(
                    filename=item_name,
                    s3_path=image_s3_path,
                    masterCategory=item["masterCategory"],
                    subCategory=item["subCategory"],
                    articleType=item["articleType"],
                    baseColour=item["baseColour"],
                    season=item["season"],
                    year=str(item["year"]),  # year might be int
                    usage=item["usage"],
                    gender=item["gender"],
                    productDisplayName=item["productDisplayName"],
                    dataset="ashraq/fashion-product-images-small",
                    created_at=datetime.datetime.now(datetime.UTC)
                ))
        logger.info(f"Processed {len(dataset)} products")

    @task.virtualenv(
        task_id="index_dataset",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def index_dataset():
        """
        Fill the product images embeddings in the database.
        """
        from airflow.models import Variable
        import mlflow
        import logging
        from domain.clip_model import ProductRetrieval
        from sqlalchemy import create_engine, MetaData, Table, select

        logger = logging.getLogger(__name__)

        logger.info("Loading the model")
        mlflow.set_tracking_uri('http://mlflow:5000')
        try:
            model_name = "fashion_product_model"
            alias = "champion"
            model_uri = f"models:/{model_name}@{alias}"
            model_champion = mlflow.pytorch.load_model(model_uri)
            logger.info("Champion model loaded successfully.")
        except Exception as e:
            logger.warning(f"Champion model not found: {e}")
            model_champion = None
        product_retrieval = ProductRetrieval(model=model_champion, bucket="prod")

        logger.info("Processing embeddings...")
        engine = create_engine(Variable.get("fashion_db_conn"))
        metadata = MetaData()
        table_fashion = Table("fashion_files", metadata, autoload_with=engine)
        try:
            with engine.connect() as conn:
                result = conn.execute(select(table_fashion))
                for row in result:
                    product_id = row['id']
                    image_path = row['s3_path']
                    embedding = product_retrieval.compute_image_embeddings(image_path)[0,:]
                    conn.execute(table_fashion.update().where(table_fashion.c.id == product_id)
                                 .values(embedding=embedding.tolist()))
        except Exception as e:
            logger.error(f"Error processing embeddings: {e}")
            raise

    get_data() >> process_dataset() >> index_dataset()


dag = recreate_prod_database()
