import datetime
from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Fashion Data

This DAG download and prepare the Fashion Product Images dataset from Hugging Face.

After preprocessing, it saves train/test splits with metadata and corresponding images into an S3 bucket.
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
    dag_id="process_etl_dataset",
    description="ETL process for Fashion dataset, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Fashion", "Dataset"],
    default_args=default_args,
    catchup=False,
)
def process_etl_dataset():
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
        import pickle as pkl
        from domain.product import Product
        from pathlib import Path
        import logging
        import io
        import datetime

        logger = logging.getLogger(__name__)

        storage_options = {"client_kwargs": {"endpoint_url": "http://s3:9000"}}
        client = boto3.client('s3')

        # Process and save training set
        logger.info("Processing training set...")
        train_dataset = load_from_disk("s3://data/interim/train_dataset", storage_options=storage_options)
        train_dir = "processed/train/"
        train_products_path = train_dir + 'product_metadata.bin'
        train_products = []
        for idx, item in enumerate(train_dataset):
            item = dict(item)  # Force conversion to dict for compatibility
            image_path = train_dir + f'product_{idx}.jpg'
            image_bytes = io.BytesIO()
            item['image'].save(image_bytes, 'JPEG')
            image_bytes.seek(0)
            client.upload_fileobj(image_bytes, "data", image_path)
            train_products.append(
                Product(product_id=idx,
                        name=item['id'],
                        description=item['productDisplayName'],
                        group=item['articleType'],
                        color=item['baseColour'],
                        master_category=item['masterCategory'],
                        image=Path("s3://data/" + image_path)))
        train_products_bytes = io.BytesIO()
        pkl.dump(train_products, train_products_bytes)
        train_products_bytes.seek(0)
        client.upload_fileobj(train_products_bytes, "data", train_products_path)
        logger.info(f"Processed {len(train_dataset)} products in training set")

        # Process and save test set
        logger.info("Processing test set...")
        test_dataset = load_from_disk("s3://data/interim/test_dataset", storage_options=storage_options)
        test_dir = "processed/test/"
        test_products_path = test_dir + 'product_metadata.bin'
        test_products = []
        for idx, item in enumerate(test_dataset):
            item = dict(item)  # Force conversion to dict for compatibility
            image_path = test_dir + f'product_{idx}.jpg'
            image_bytes = io.BytesIO()
            item['image'].save(image_bytes, 'JPEG')
            image_bytes.seek(0)
            client.upload_fileobj(image_bytes, "data", image_path)
            test_products.append(
                Product(product_id=idx,
                        name=item['id'],
                        description=item['productDisplayName'],
                        group=item['articleType'],
                        color=item['baseColour'],
                        master_category=item['masterCategory'],
                        image=Path("s3://data/" + image_path)))
        test_products_bytes = io.BytesIO()
        pkl.dump(test_products, test_products_bytes)
        test_products_bytes.seek(0)
        client.upload_fileobj(test_products_bytes, "data", test_products_path)
        logger.info(f"Processed {len(test_dataset)} products in test set")

        # Track the experiment
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Fashion Product Dataset")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Fashion Product Dataset"},
                         log_system_metrics=True)
        mlflow.log_param("Train observations", len(train_products))
        mlflow.log_param("Test observations", len(test_products))
        mlflow.log_param("Train products path", "s3://data/" + train_dir)
        mlflow.log_param("Test products path", "s3://data/" + test_dir)

    get_data() >> split_dataset() >> process_datasets()

dag = process_etl_dataset()
