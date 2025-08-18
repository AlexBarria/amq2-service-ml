"""
REST API for ML-based fashion product retrieval.

This module defines a FastAPI application that exposes endpoints for searching similar fashion products
by text description or image. It communicates with a gRPC model service for inference and uses S3 for image storage.
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
import grpc
import ml_service_pb2
import ml_service_pb2_grpc
import boto3
import io
import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()


class DescriptionRequest(BaseModel):
    """
    Request model for searching by product description.

    Attributes:
        description (str): The product description to search for.
    """
    description: str

class S3Request(BaseModel):
    """
    Request model for processing a file already in S3.

    Attributes:
        bucket (str): The S3 bucket where the file is located.
        key (str): The object key (path) of the file in the bucket.
    """
    bucket: str
    key: str

@app.get("/")
def read_root():
    """
    Health check endpoint.

    Returns:
        dict: A message indicating the REST ML Service is running.
    """
    return {"message": "REST ML Service is running. Post to /search/description or /search/image"}


@app.post("/search/description")
async def search_by_description(request: DescriptionRequest):
    """
    Searches for similar fashion products using a text description.

    Args:
        request (DescriptionRequest): The request body containing the description.

    Returns:
        list[dict]: List of similar fashion products as dictionaries.
    """
    logging.info("Received search request with description: %s", request.description)
    description = request.description
    with grpc.insecure_channel("model_grpc:50051") as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)
        request = ml_service_pb2.Search(description=description, image_path=None)
        response = stub.Predict(request)
    logger.info(f"gRPC response received for description: {description}")
    return [
        {
            "id": prod.id,
            "filename": prod.filename,
            "s3_path": prod.s3_path,
            "masterCategory": prod.master_category,
            "subCategory": prod.sub_category,
            "articleType": prod.article_type,
            "baseColour": prod.base_colour,
            "season": prod.season,
            "year": prod.year,
            "usage": prod.usage,
            "gender": prod.gender,
            "productDisplayName": prod.product_display_name,
            "dataset": prod.dataset,
            "created_at": prod.created_at
        }
        for prod in response.fashion_product
    ]


@app.post("/search/image")
async def search_by_image(request: Request):
    """
    Searches for similar fashion products by uploading an image.

    Args:
        request (Request): The raw HTTP request containing the image binary.

    Returns:
        list[dict]: List of similar fashion products as dictionaries.
    """
    logging.info("Received search request with image")
    image_bytes = await request.body()
    s3_client = boto3.client('s3')
    s3_dir = "fashion_files/"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    item_name = f"{timestamp}.jpg"
    image_s3_path = s3_dir + item_name
    image_fileobj = io.BytesIO(image_bytes)
    s3_client.upload_fileobj(image_fileobj, "tmp", image_s3_path)
    logging.info("Image uploaded to S3 at path: %s", image_s3_path)
    with grpc.insecure_channel("model_grpc:50051") as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)
        request = ml_service_pb2.Search(description=None, image_path=image_s3_path)
        response = stub.Predict(request)
    logger.info(f"gRPC response received for image: {image_s3_path}")
    return [
        {
            "id": prod.id,
            "filename": prod.filename,
            "s3_path": prod.s3_path,
            "masterCategory": prod.master_category,
            "subCategory": prod.sub_category,
            "articleType": prod.article_type,
            "baseColour": prod.base_colour,
            "season": prod.season,
            "year": prod.year,
            "usage": prod.usage,
            "gender": prod.gender,
            "productDisplayName": prod.product_display_name,
            "dataset": prod.dataset,
            "created_at": prod.created_at
        }
        for prod in response.fashion_product
    ]


@app.post("/predict")
async def predict_from_s3(request: S3Request):
    """
    Triggers a prediction for an image already stored in S3.
    This is called by the Kafka dispatcher.

    Args:
        request (S3Request): The request body containing the bucket and key.

    Returns:
        list[dict]: List of similar fashion products as dictionaries.
    """
    # Combine the bucket and key into a single path
    image_s3_path = f"{request.bucket}/{request.key}"

    logging.info("Received prediction request for S3 object: %s", image_s3_path)

    with grpc.insecure_channel("model_grpc:50051") as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)
        grpc_request = ml_service_pb2.Search(description=None, image_path=image_s3_path)
        response = stub.Predict(grpc_request)

    logger.info("gRPC response received for S3 object: %s", image_s3_path)
    return [
        {
            "id": prod.id,
            "filename": prod.filename,
            "s3_path": prod.s3_path,
            "masterCategory": prod.master_category,
            "subCategory": prod.sub_category,
            "articleType": prod.article_type,
            "baseColour": prod.base_colour,
            "season": prod.season,
            "year": prod.year,
            "usage": prod.usage,
            "gender": prod.gender,
            "productDisplayName": prod.product_display_name,
            "dataset": prod.dataset,
            "created_at": prod.created_at
        }
        for prod in response.fashion_product
    ]