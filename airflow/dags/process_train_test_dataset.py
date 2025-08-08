import datetime
from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Train/Test Fashion Dataset

This DAG download and prepare the Fashion Product Images dataset from Hugging Face.

After preprocessing, it saves train/test splits and saves metadata in SQL DB and corresponding images into an S3 bucket.
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
    dag_id="process_train_test_dataset",
    description="ETL process for Fashion dataset, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Fashion", "Dataset"],
    default_args=default_args,
    catchup=False,
)
def process_train_test_dataset():
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
        max_products = Variable.get("max_products", default_var=None)

        if max_products and max_products < len(dataset):
            dataset = dataset.select(range(max_products))
            logger.info(f'Subset of {max_products} selected from the dataset')

        # Save information of the dataset
        storage_options = {"client_kwargs": {"endpoint_url": "http://s3:9000"}}
        dataset.save_to_disk("s3://data/raw/fashion-product", storage_options=storage_options)
        logger.info('Dataset saved in s3')

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        from datasets import load_from_disk
        from airflow.models import Variable
        import logging

        logger = logging.getLogger(__name__)

        storage_options = {"client_kwargs": {"endpoint_url": "http://s3:9000"}}
        dataset = load_from_disk("s3://data/raw/fashion-product", storage_options=storage_options)

        test_size = Variable.get("test_size")

        # Shuffle and split dataset
        dataset = dataset.shuffle(seed=42)
        split_idx = int(len(dataset) * (1 - test_size))
        train_dataset = dataset.select(range(split_idx))
        test_dataset = dataset.select(range(split_idx, len(dataset)))
        logger.info(f"Dataset split: {len(train_dataset)} train / {len(test_dataset)} test")

        # Save splits
        train_dataset.save_to_disk("s3://data/interim/train_dataset", storage_options=storage_options)
        test_dataset.save_to_disk("s3://data/interim/test_dataset", storage_options=storage_options)
        logger.info('Dataset train/test split in s3')

    @task.virtualenv(
        task_id="process_datasets",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def process_datasets():
        """
        Process the training and test dataset by creating the products metadata and saving images.
        """
        from datasets import load_from_disk
        from airflow.models import Variable
        import mlflow
        import boto3
        import logging
        import io
        import datetime
        from sqlalchemy import create_engine, Column, Integer, String, DateTime, MetaData, Table

        logger = logging.getLogger(__name__)

        storage_options = {"client_kwargs": {"endpoint_url": "http://s3:9000"}}
        s3_client = boto3.client('s3')

        engine = create_engine(Variable.get("fashion_db_conn"))
        metadata = MetaData()
        table_train = Table("train_dataset", metadata,
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
                            )
        table_test = Table("test_dataset", metadata,
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
                           )
        metadata.drop_all(engine, checkfirst=True)
        metadata.create_all(engine)

        # Process and save training set
        logger.info("Processing training set...")
        train_dataset = load_from_disk("s3://data/interim/train_dataset", storage_options=storage_options)
        train_s3_dir = "processed/train/"
        for idx, item in enumerate(train_dataset):
            item = dict(item)  # Force conversion to dict for compatibility
            item_name =  f"{idx}_{item['id']}.jpg"
            image_s3_path = train_s3_dir + item_name
            image_bytes = io.BytesIO()
            item['image'].save(image_bytes, 'JPEG')
            image_bytes.seek(0)
            s3_client.upload_fileobj(image_bytes, "data", image_s3_path)
            with engine.begin() as conn:
                conn.execute(table_train.insert().values(
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
        logger.info(f"Processed {len(train_dataset)} products in training set")

        # Process and save test set
        logger.info("Processing test set...")
        test_dataset = load_from_disk("s3://data/interim/test_dataset", storage_options=storage_options)
        test_s3_dir = "processed/test/"
        for idx, item in enumerate(test_dataset):
            item = dict(item)  # Force conversion to dict for compatibility
            item_name = f"{idx}_{item['id']}.jpg"
            image_s3_path = test_s3_dir + item_name
            image_bytes = io.BytesIO()
            item['image'].save(image_bytes, 'JPEG')
            image_bytes.seek(0)
            s3_client.upload_fileobj(image_bytes, "data", image_s3_path)
            with engine.begin() as conn:
                conn.execute(table_test.insert().values(
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
        logger.info(f"Processed {len(test_dataset)} products in test set")

        # Track the experiment
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Fashion Product Dataset")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Fashion Product Dataset"},
                         log_system_metrics=True)
        mlflow.log_param("Train observations", len(train_dataset))
        mlflow.log_param("Test observations", len(test_dataset))
        mlflow.log_param("Train items metadata table", table_train.name)
        mlflow.log_param("Train items image path", "s3://data/" + train_s3_dir)
        mlflow.log_param("Test items metadata table", table_test.name)
        mlflow.log_param("Test items image path", "s3://data/" + test_s3_dir)

    get_data() >> split_dataset() >> process_datasets()

dag = process_train_test_dataset()
