import logging
from typing import List
from PIL import Image
from torch.utils.data import Dataset

from domain.product import Product

logger = logging.getLogger(__name__)


class FashionDataset(Dataset):
    """
    A PyTorch Dataset for the Fashion Product Images dataset from Hugging Face.
    This dataset wraps a Product dataset and applies CLIP preprocessing
    and tokenization.

    Args:
        product_dataset (List[Product]): The product dataset to wrap.
        preprocess (callable): A preprocessing function for images, typically
            from CLIP.
        tokenizer (callable): A tokenizer function for text, typically from
            CLIP.
        preload (bool, optional): If true, preloads the images to improve
            training performance.
    """

    def __init__(
            self,
            product_dataset: List[Product],
            preprocess: callable,
            tokenizer: callable,
            preload: bool = True,
    ):
        self.dataset = product_dataset
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.preload = preload
        self.images = []
        if preload:
            for product in self.dataset:
                self.images.append(Image.open(product.image).convert('RGB'))

    def __len__(self) -> int:
        """ Returns the number of items in the dataset. """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: (image_tensor, text_tensor) after preprocessing/tokenization.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If expected keys are missing from the sample.
        """
        try:
            item = self.dataset[idx]
            if self.preload:
                image = self.images[idx]
            else:
                image = Image.open(self.dataset[idx].image).convert('RGB')
            text = item.description

            image_tensor = self.preprocess(image)
            text_tensor = self.tokenizer(text)

            return image_tensor, text_tensor

        except KeyError as e:
            logger.error(f"Missing key {e} at index {idx}. Check dataset structure.")
            raise

        except Exception as e:
            logger.error(f"Error processing item at index {idx}: {e}")
            raise
