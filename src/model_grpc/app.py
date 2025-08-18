"""
gRPC server for ML-based fashion product retrieval.

This module defines the gRPC service for product search using text or image queries.
It loads a CLIP-based model, computes embeddings, and retrieves similar products from a PostgreSQL database.
"""
import grpc
from concurrent import futures

from sqlalchemy import create_engine, MetaData, Table
import logging
import mlflow

import ml_service_pb2
import ml_service_pb2_grpc
import os
from domain.clip_model import ProductRetrieval
from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    """
    gRPC service for fashion product retrieval.
    """

    def __init__(self, model=None):
        """
        Initializes the MLServiceServicer.

        Args:
            model (ProductRetrieval): The product retrieval engine.
        """
        self.model = model

    def Predict(self, request, context):
        """
        Handles prediction requests.

        Computes embeddings for the provided description or image path,
        retrieves the most similar products from the database, and returns them.

        Args:
            request (ml_service_pb2.Search): The gRPC request containing description or image_path.
            context: gRPC context.

        Returns:
            ml_service_pb2.Prediction: The prediction response with similar products.
        """
        if self.model is None:
            logger.error("Model is unavailable.")
            context.set_details("Model unavailable")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return ml_service_pb2.Prediction()

        description = request.description
        image_path = request.image_path

        if description is not None and len(description) > 0:
            logger.info(f"Computing text embedding for description: {description}")
            embedding = self.model.compute_text_embeddings(description)[0, :]
        elif image_path is not None and len(image_path) > 0:
            logger.info(f"Computing image embedding for image_path: {image_path}")

            # Check if the path contains a bucket/key structure
            if '/' in image_path:
                # If it does, split it into bucket and key
                bucket, key = image_path.split('/', 1)
            else:
                # Otherwise, use the default bucket ("tmp") from the model and the full path as the key
                bucket = self.model.bucket
                key = image_path
            
            logger.info(f"Attempting to download object '{key}' from bucket '{bucket}'")
            
            # Call the model function with both bucket and key
            embedding = self.model.compute_image_embeddings(bucket, key)[0, :]

        else:
            logger.error("No valid input provided for prediction.")
            context.set_details("No valid input provided for prediction.")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ml_service_pb2.Prediction()
        try:
            engine = create_engine(os.getenv("PG_CONN_STR"))
            metadata = MetaData()
            table_fashion = Table("fashion_files", metadata, autoload_with=engine)
            with engine.connect() as conn:
                result = conn.execute(table_fashion.select()
                                    .order_by(table_fashion.c.embedding.cosine_distance(embedding)).limit(5))
            products = []
            for row in result.mappings():
                product = ml_service_pb2.FashionProduct(
                    id=row['id'],
                    filename=row['filename'],
                    s3_path=row['s3_path'],
                    master_category=row['masterCategory'],
                    sub_category=row['subCategory'],
                    article_type=row['articleType'],
                    base_colour=row['baseColour'],
                    season=row['season'],
                    year=row['year'],
                    usage=row['usage'],
                    gender=row['gender'],
                    product_display_name=row["productDisplayName"],
                    dataset=row['dataset'],
                    created_at=str(row['created_at']),
                    embedding=list(row['embedding']) if hasattr(row, "embedding") else [])
                products.append(product)
            return ml_service_pb2.Prediction(fashion_product=products)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            context.set_details("Prediction failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return ml_service_pb2.Prediction()


def serve():
    """
    Loads the model and starts the gRPC server.

    Loads the champion model from MLflow, initializes the product retrieval engine,
    and starts the gRPC server to listen for prediction requests.
    """
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
    product_retrieval = ProductRetrieval(model=model_champion, bucket="tmp")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(product_retrieval), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Servidor gRPC escuchando en el puerto 50051...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
