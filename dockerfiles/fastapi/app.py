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
    description: str


@app.get("/")
def read_root():
    return {"message": "REST ML Service is running. Post to /search/description or /search/image"}


@app.post("/search/description")
async def search_by_description(request: DescriptionRequest):
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