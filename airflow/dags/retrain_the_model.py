import datetime
import logging
from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

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
    'dagrun_timeout': datetime.timedelta(minutes=15)
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
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def finetune_the_challenger_model():
        from datasets import load_from_disk
        import mlflow
        import torch
        from domain.dataset import FashionDataset
        from domain.trainer import CLIPFineTuner
        from torch.utils.data import DataLoader

        mlflow.set_tracking_uri('http://mlflow:5000')

        def collate_fn(batch):
            """
            Custom collate function to stack image and text tensors for DataLoader.

            Args:
                batch (list): A list of (image, text) tuples.

            Returns:
                images: Tensor of shape [batch_size, 3, 224, 224]
                texts:  Tensor of shape [batch_size, 77]
            """
            images, texts = zip(*batch)
            images = torch.stack(images)
            texts = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in texts], dim=0)
            return images, texts

        logger.info("Fine-tuning CLIP")
        storage_options = {"client_kwargs": {"endpoint_url": "http://localhost:9000"}}
        dataset = load_from_disk("s3://data/processed/train/product_metadata.bin", storage_options=storage_options)
        try:
            trainer = CLIPFineTuner(model_name="ViT-B-32", pretrained="openai")
        except Exception as e:
            logger.error(f"Failed to initialize CLIPFineTuner: {e}")
            raise

        preprocess = trainer.get_preprocessor()
        tokenizer = trainer.get_tokenizer()

        # Wrap dataset for PyTorch
        try:
            custom_dataset = FashionDataset(
                product_dataset=dataset,
                preprocess=preprocess,
                tokenizer=tokenizer,
            )
        except Exception as e:
            logger.error(f"Failed to create FashionDataset: {e}")
            raise

        # Create DataLoader
        try:
            dataloader = DataLoader(
                custom_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
            )
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise

        # Configure optimizer and train
        trainer.configure_optimizer(lr=1.0e-5)
        try:
            training_logs = trainer.train(dataloader, num_epochs=10)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return

    finetune_the_challenger_model()


my_dag = finetune()
