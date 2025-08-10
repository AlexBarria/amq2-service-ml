import datetime
from airflow.decorators import dag, task

markdown_text = """
### Fine tune the CLIP model on a Fashion Product Images dataset.

This DAG fine-tune the model based on new data, tests the previous model, and put in production the new one if it 
performs  better than the old one.
"""

default_args = {
    'owner': "Alex Barria, Clara Bureu, Maximiliano Torti",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=30)
}


@dag(
    dag_id="finetune",
    description="Fine tune the model based on new data, tests the previous model, and put in production the new one if "
                "it performs better than the old one",
    doc_md=markdown_text,
    tags=["Fine tune", "Fashion", "Model"],
    default_args=default_args,
    catchup=False,
)
def finetune():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def finetune_the_challenger_model():
        """
        Fine-tunes the CLIP model on the latest fashion product dataset.

        Loads training data from the database, prepares the dataset and dataloader,
        fine-tunes the model, and logs the new model as a 'challenger' in MLflow.
        Tracks training parameters and artifacts for experiment management.
        """
        import mlflow
        from airflow.models import Variable
        from domain.dataset import FashionDataset
        from domain.trainer import CLIPFineTuner
        from domain.utils import collate_fn
        from domain.product import Product
        from torch.utils.data import DataLoader
        import logging
        import datetime
        from sqlalchemy import create_engine, MetaData, Table, select
        from pathlib import Path

        logger = logging.getLogger(__name__)

        logger.info("Fine-tuning CLIP")
        logger.info("Loading pre-trained model")
        try:
            trainer = CLIPFineTuner(model_name="ViT-B-32", pretrained="openai")
        except Exception as e:
            logger.error(f"Failed to initialize CLIPFineTuner: {e}")
            raise
        preprocess = trainer.get_preprocessor()
        tokenizer = trainer.get_tokenizer()

        logger.info("Loading train dataset")
        engine = create_engine(Variable.get("fashion_db_conn"))
        metadata = MetaData()
        table_train = Table("train_dataset", metadata, autoload_with=engine)
        try:
            with engine.connect() as conn:
                result = conn.execute(select(table_train))
                train_records = [dict(row) for row in result]
            train_products = []
            for record in train_records:
                product = Product(
                    product_id=record["id"],
                    name=record["filename"],
                    description=record['productDisplayName'],
                    group=record['articleType'],
                    color=record['baseColour'],
                    master_category=record['masterCategory'],
                    image=Path(record["s3_path"]))
                train_products.append(product)
            custom_dataset = FashionDataset(
                product_dataset=train_products,
                preprocess=preprocess,
                tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Failed to create FashionDataset: {e}")
            raise
        logger.info("Creating dataloader")
        batch_size = Variable.get("batch_size", default_var=None)
        try:
            dataloader = DataLoader(
                custom_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
            )
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise

        # Configure optimizer and train
        logger.info("Start finetuning")
        lr = Variable.get("lr", default_var=None)
        num_epochs = Variable.get("num_epochs", default_var=None)
        trainer.configure_optimizer(lr=lr)
        try:
            training_logs = trainer.train(dataloader, num_epochs=num_epochs)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return
        logger.info("Finetuning completed")

        # Track the experiment
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Fashion Product Dataset")

        mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "challenger models", "dataset": "Fashion Product Dataset"},
                         log_system_metrics=True)
        artifact_path = "model"
        model_name = "fashion_product_model"
        try:
            logger.info("MLFlow logging model params")
            params = {
                "model": "ViT-B-32",
                "pretrained": "openai",
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_epochs": num_epochs,
                'training_logs': training_logs
            }
            mlflow.log_params(params)
            tags = {
                "model": "ViT-B-32",
                "pretrained": "openai",
                "dataset": "Fashion Product Dataset",
                "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            mlflow.pytorch.log_model(
                pytorch_model=trainer.model,
                artifact_path=artifact_path,
                registered_model_name=model_name
            )
            mlflow_client = mlflow.MlflowClient()
            model_info = mlflow_client.get_latest_versions(model_name, stages=["None"])[0]
            for tag_key, tag_value in tags.items():
                mlflow_client.set_model_version_tag(
                    name=model_name,
                    version=model_info.version,
                    key=tag_key,
                    value=tag_value
                )
            try:
                mlflow_client.delete_registered_model_alias(model_name, "challenger")
            except:
                logger.warning("Challenger alias not found, proceeding to set it.")
            mlflow_client.set_registered_model_alias(model_name, "challenger", model_info.version)
            logger.info("Model params logged")
        except Exception as e:
            logger.error(f"Failed to log model params: {e}")

    @task.virtualenv(
        task_id="evaluate_challenger_model",
        requirements=["datasets==3.6.0"],
        system_site_packages=True
    )
    def evaluate_champion_challenge():
        """
        Evaluates the performance of the 'champion' and 'challenger' models.

        Loads the test dataset and both models from MLflow, computes top-3 description accuracy,
        and promotes the challenger to champion if it outperforms the current champion.
        """
        import mlflow
        from airflow.models import Variable
        from domain.product import Product
        from domain.clip_model import ProductRetrieval
        from domain.metrics import top_k_description_accuracy_score
        import logging
        from sqlalchemy import create_engine, MetaData, Table, select
        from pathlib import Path


        logger = logging.getLogger(__name__)

        mlflow.set_tracking_uri('http://mlflow:5000')

        logger.info("Loading test dataset")
        engine = create_engine(Variable.get("fashion_db_conn"))
        metadata = MetaData()
        table_test = Table("test_dataset", metadata, autoload_with=engine)
        try:
            with engine.connect() as conn:
                result = conn.execute(select(table_test))
                train_records = [dict(row) for row in result]
            test_products = []
            for record in train_records:
                product = Product(
                    product_id=record["id"],
                    name=record["filename"],
                    description=record['productDisplayName'],
                    group=record['articleType'],
                    color=record['baseColour'],
                    master_category=record['masterCategory'],
                    image=Path(record["s3_path"]))
                test_products.append(product)
        except Exception as e:
            logger.error(f"Failed to create FashionDataset: {e}")
            raise

        # Load the champion model
        logger.info("Loading champion model")
        model_name = "fashion_product_model"
        try:
            alias = "champion"
            model_uri = f"models:/{model_name}@{alias}"
            model_champion = mlflow.pytorch.load_model(model_uri)
            logger.info("Champion model loaded successfully.")
        except Exception as e:
            logger.warning(f"Champion model not found: {e}")
            model_champion = None

        # Testing metrics of the champion model
        logger.info("Testing champion model")
        product_retrieval = ProductRetrieval(model=model_champion, bucket="data")
        product_retrieval.index_product_database(test_products)
        champion_accuracy = top_k_description_accuracy_score(product_retrieval, test_products, k=3)
        logger.info(f"Top-3 description champion accuracy: {champion_accuracy}")

        # Load the challenger model
        logger.info("Loading challenger model")
        alias = "challenger"
        model_uri = f"models:/{model_name}@{alias}"
        challenger_model = mlflow.pytorch.load_model(model_uri)
        product_retrieval = ProductRetrieval(model=challenger_model,bucket="data")
        product_retrieval.index_product_database(test_products)
        challenger_accuracy = top_k_description_accuracy_score(product_retrieval, test_products, k=3)
        logger.info(f"Top-3 description challenger accuracy: {challenger_accuracy}")

        if challenger_accuracy > champion_accuracy:
            logger.info("Challenger model outperforms champion model. Promoting challenger to champion.")
            mlflow_client = mlflow.MlflowClient()
            try:
                mlflow_client.delete_registered_model_alias(model_name, "champion")
                challenger_version = mlflow_client.get_model_version_by_alias(model_name, "challenger")
                mlflow_client.delete_registered_model_alias(model_name, "challenger")
                mlflow_client.set_registered_model_alias(model_name, "champion", challenger_version.version)
                logger.info("Challenger model promoted to champion.")
            except:
                logger.error("Failed to promote challenger to champion.")
        else:
            logger.info("Challenger model does not outperform champion model")

    finetune_the_challenger_model() >> evaluate_champion_challenge()


my_dag = finetune()
