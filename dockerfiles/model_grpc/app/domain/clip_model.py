"""
CLIP-based product similarity engine using FAISS indexing for retrieval.
Supports both text and image queries.
Classes:
    - ProductRetrieval: Encapsulates model loading, embedding generation, and FAISS indexing.
"""
import logging
import faiss
import numpy as np
import open_clip
import torch
from pathlib import Path
from typing import List, Union
from PIL import Image
from domain.product import Product
import boto3
import io

logger = logging.getLogger(__name__)


class ProductRetrieval:
    def __init__(self, model, bucket: str = "data"):
        """
        Initializes the CLIP model and processor for embedding generation.
        If model is not provided, it uses the default openai pretrained model.

        Args:
            model (torch.nn.Module): CLIP model.
            bucket (str): S3 bucket name where images are stored.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        if model:
            _, _, self.processor = open_clip.create_model_and_transforms('ViT-B-32')
            self.model = model
            logger.info(f"Initialized CLIP model from fine-tuned")
        else:
            self.model, _, self.processor = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            logger.info("Initialized CLIP model from pretrained openai")
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.index = None
        self.index_products = []
        self.bucket = bucket

    def compute_text_embeddings(self, text: str) -> np.ndarray:
        """
        Computes text embeddings for a product description.

        Args:
            text (str): a product description.

        Returns:
            np.ndarray: Text embeddings.
        """
        try:
            inputs = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(inputs)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            logger.debug(f"Text embedding computed for query: {text}")
            return text_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to compute text embeddings: {e}")
            raise

    def compute_image_embeddings(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Computes image embeddings for a single image.

        Args:
            image_path (str | Path ): Path to image.

        Returns:
            np.ndarray: Normalized image embeddings.
        """
        try:
            image = None
            if isinstance(image_path, (str, Path)):
                try:
                    client = boto3.client('s3')
                    image_bytes = io.BytesIO()
                    client.download_fileobj(self.bucket, str(image_path), image_bytes)
                    image_bytes.seek(0)
                    image = Image.open(image_bytes).convert('RGB')
                except Exception as e:
                    logger.error(f"Failed to open image {image}: {e}")
                    raise ValueError(f"Invalid image path or format: {image}")

            inputs = self.processor(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(inputs)
                # Embeddings normalization for cosine similarity
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            logger.debug(f"Image embedding computed for image: {image}")
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to compute image embeddings: {e}")
            raise

    def index_product_database(self, products: List[Product]):
        """
        Indexes a list of products using their image embeddings into a FAISS index.

        Args:
            products (List[Product]): List of product objects with image paths.
        """
        logger.info(f"Indexing {len(products)} products...")
        self.index_products = products
        product_paths = [str(product.image) for product in products]

        # Get a sample embedding to determine the dimension and initialize the FAISS index
        sample_embedding = self.compute_image_embeddings(product_paths[0])
        dimension = sample_embedding.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        batch_size = 32
        for i in range(0, len(product_paths), batch_size):
            batch_paths = product_paths[i:i + batch_size]
            batch_embeddings = []

            for path in batch_paths:
                embedding = self.compute_image_embeddings(path)
                batch_embeddings.append(embedding)

            # Concat the embeddings of the batch and add to the index
            batch_embeddings = np.vstack(batch_embeddings)
            self.index.add(batch_embeddings)
            logger.debug(f"Indexed batch {i}–{i + len(batch_paths) - 1}")

    def find_similar_products(self, query: Union[str, Path, Image.Image], top_k: int = 5) -> dict:
        """
        Finds the most similar products to the given query (image or text) using FAISS.

        Args:
            query (str | Path | Image): Text string or image path/object.
            top_k (int): Number of results to return.

        Returns:
            dict: Query info and list of matched products with similarity scores.
        """
        if self.index is None:
            raise ValueError("Product index is not initialized. Call index_product_database() first.")

        # Determine if the query is text or image
        if isinstance(query, str) and not Path(query).exists():
            query_embedding = self.compute_text_embeddings(query)
            query_type = 'text'
        else:
            query_embedding = self.compute_image_embeddings(query)
            query_type = 'image'

        k = min(top_k, len(self.index_products))
        scores, indices = self.index.search(query_embedding, k)
        logger.debug(f"Query type: {query_type} | Top-{k} matches retrieved")

        results = {"query": query,
                   "query_type": query_type,
                   "matches": []}
        for score, idx in zip(scores[0], indices[0]):
            results["matches"].append((self.index_products[idx].to_json(), float(score)))

        return results
