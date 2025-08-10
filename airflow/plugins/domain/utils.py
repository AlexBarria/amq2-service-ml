import torch
import logging

logger = logging.getLogger(__name__)


# NOTE: Kept in utils for now. Move into dataset class only if batching logic becomes dataset-specific.
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
