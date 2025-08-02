import os
import boto3
import requests
from datasets import load_dataset
from sqlalchemy import create_engine, Column, Integer, String, DateTime, MetaData, Table
from sqlalchemy.exc import OperationalError
from botocore.client import Config
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class FashionDatasetHandler:
    def __init__(self, name: str, split: str = "train", limit: int = 100, output_dir: str = "images"):
        self.name = name
        self.split = split
        self.limit = limit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = load_dataset(name, split=f"{split}[:{limit}]")

    def download_images(self) -> list[dict]:
        print(f"[↓] Downloading {self.limit} images from {self.name}...")
        records = []
        for i, item in enumerate(self.dataset):
            url = item["image"]["path"] if isinstance(item["image"], dict) else item["image"]
            image_bytes = item["image"].convert("RGB").tobytes()
            filename = f"{i}_{item['id']}.jpg"
            local_path = self.output_dir / filename
            item["local_path"] = local_path
            item["filename"] = filename
            item["dataset"] = self.name

            # Save image
            item["image"].save(local_path)
            records.append(item)
        print(f"[✓] Downloaded and saved {len(records)} images.")
        return records


class S3Uploader:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="us-east-1"
        )
        self.ensure_bucket_exists()

    def ensure_bucket_exists(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
            print(f"[✓] Bucket '{self.bucket}' already exists.")
        except self.client.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                print(f"[+] Bucket '{self.bucket}' does not exist. Creating...")
                self.client.create_bucket(Bucket=self.bucket)
            elif error_code == 403:
                print(f"[!] Bucket '{self.bucket}' exists but access is forbidden. Check credentials or bucket ownership.")
                raise
            else:
                raise

    def upload_file(self, local_path: Path, s3_key: str):
        self.client.upload_file(str(local_path), self.bucket, s3_key)
        return s3_key


class MetadataIndexer:
    def __init__(self):
        self.engine = create_engine(os.getenv("PG_CONN_STR"))
        self.metadata = MetaData()
        self.table = Table(
            "fashion_files", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("filename", String),
            Column("s3_path", String),
            Column("category", String),
            Column("gender", String),
            Column("productDisplayName", String),
            Column("dataset", String),
            Column("created_at", DateTime),
        )
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        try:
            self.metadata.create_all(self.engine)
        except OperationalError as e:
            print(f"[x] Error creating table: {e}")
            raise

    def insert_record(self, record: dict, s3_path: str):
        with self.engine.begin() as conn:
            conn.execute(self.table.insert().values(
                filename=record["filename"],
                s3_path=s3_path,
                category=record.get("masterCategory", "unknown"),
                gender=record.get("gender", "unknown"),
                productDisplayName=record.get("productDisplayName", ""),
                dataset=record.get("dataset", ""),
                created_at=datetime.utcnow()
            ))


def main():
    dataset_name = "ashraq/fashion-product-images-small"
    s3_bucket = os.getenv("S3_BUCKET", "hf-datasets")
    s3_prefix = os.getenv("S3_PREFIX", "fashion")

    # Step 1: Download images
    handler = FashionDatasetHandler(dataset_name, limit=100)
    records = handler.download_images()

    # Step 2: Upload to S3
    uploader = S3Uploader(bucket=s3_bucket)
    indexer = MetadataIndexer()

    for record in records:
        s3_key = f"{s3_prefix}/{record['filename']}"
        uploader.upload_file(record["local_path"], s3_key)
        indexer.insert_record(record, s3_key)
        print(f"[✓] Uploaded and indexed {record['filename']}")


if __name__ == "__main__":
    main()