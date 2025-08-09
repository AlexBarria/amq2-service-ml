import grpc
from concurrent import futures

from sqlalchemy import create_engine, MetaData, Table, select
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
    def __init__(self, model=None):
        self.model = model

    def Predict(self, request, context):
        if self.model is None:
            logger.error("Model is unavailable.")
            context.set_details("Model unavailable")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return ml_service_pb2.Prediction()

        description = request.description
        image_path = request.image_path

        if description is not None and len(description) > 0:
            engine = create_engine(os.getenv("PG_CONN_STR"))
            metadata = MetaData()
            table_fashion = Table("fashion_files", metadata, autoload_with=engine)
            embedding = self.model.compute_text_embeddings(description)[0, :]
            try:
                with engine.connect() as conn:
                    result = conn.execute(select(table_fashion)
                                          .order_by(table_fashion.c.embedding.cosine_distance(embedding)).limit(5))
            except Exception as e:
                logger.error(f"Read execute failed: {e}")
            try:
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

        logger.error(f"Unimplemented")
        context.set_details("Unimplemented")
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return ml_service_pb2.Prediction()


def serve():
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

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(product_retrieval), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Servidor gRPC escuchando en el puerto 50051...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
