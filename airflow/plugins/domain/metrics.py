"""
Calculate metrics for fashion similarity retrieval model.
"""
import logging
import warnings
from typing import List

from domain.product import Product
from domain.clip_model import ProductRetrieval

logger = logging.getLogger(__name__)


def top_k_description_accuracy_score(model: ProductRetrieval, products: List[Product], k=5):
    """
    Calculate the top-k retrieval accuracy metric for detailed descriptions.
    Given a detailed description, it evaluates whether the model can retrieve the image to which the description belongs
    among the top k predictions.

    Args:
        model (ProductRetrieval): Product retrieval model.
        products (List[Product]): Dictionary containing the products id, description and the product image path.
        k (int): top-k number of predictions for similar products retrieval.

    Returns:
        float: Top-k retrieval accuracy score for detailed descriptions.
    """
    if not products:
        warnings.warn("No products provided to evaluate description accuracy.", UserWarning)
        return 0.0

    correct = 0
    for product in products:
        query = product.description
        true_id = product.id
        try:
            predictions = model.find_similar_products(query, top_k=k)
            prediction_ids = [match[0]['id'] for match in predictions['matches']]
            if true_id in prediction_ids:
                correct += 1
            logger.debug(f"Query '{query}' matched: {true_id in prediction_ids}")
        except Exception as e:
            logger.error(f"Error processing product {product.id} with description '{product.description}': {e}")

    return correct / len(products)