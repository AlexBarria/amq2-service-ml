import os
import logging
import warnings
import torch
import torch.nn.functional as F
import open_clip

logger = logging.getLogger(__name__)


class CLIPFineTuner:
    """
    Fine-tunes a CLIP model for image-text matching tasks.
    This class provides methods to configure the model, preprocess images,
    tokenize text, compute contrastive loss, and train the model on a dataset.
    """

    def __init__(
            self,
            model_name: str = "ViT-B-32",
            pretrained: str = "openai",
            device: str = None,
            model: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None
    ):
        """
        Initialize the CLIPFineTuner.

        Args:
            model_name (str, optional): Name of the CLIP model architecture
                to use. Defaults to "ViT-B-32".
            pretrained (str, optional): Name of the pretrained weights to
                load. Defaults to "openai".
            device (str, optional): Device to use for computation ("cuda",
                "cpu", etc.). If None, automatically selects CUDA if available.
            model (torch.nn.Module, optional): Existing CLIP model instance to
                use. If None, a new model is created.
            optimizer (torch.optim.Optimizer, optional): Optimizer instance
                for training. If None, must be configured later.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: check if there's a better way to handle preprocess
        # Always get preprocess from open_clip
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        if model is not None and optimizer is not None:
            self.model = model.to(self.device)
            self.optimizer = optimizer
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained
            )
            self.model = self.model.to(self.device)
            self.optimizer = None
            warnings.warn("Optimizer not provided; call configure_optimizer() before training.", UserWarning)
            logger.warning("Optimizer not provided at init; training will fail if not configured later.")

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.loss_history = []

    def get_preprocessor(self) -> callable:
        """
        Get the preprocessing function for the CLIP model.
        Returns:
            Callable: Preprocessing function that takes an image and returns a tensor.
        """
        return self.preprocess

    def get_tokenizer(self) -> callable:
        """
        Get the tokenizer for the CLIP model.
        Returns:
            Tokenizer: The tokenizer used for text processing in CLIP.
        """
        return self.tokenizer

    def configure_optimizer(self,
                            lr: float = 1e-5,
                            weight_decay: float = 1e-4) -> None:
        """
        Configure the optimizer for fine-tuning the CLIP model.
        Args:
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for regularization.
        Returns:
            None
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=lr,
                                           weight_decay=weight_decay)
        logger.info(f"Optimizer configured: AdamW | lr={lr}, weight_decay={weight_decay}")

    def contrastive_loss(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute the contrastive loss between image and text features.
        Args:
            image_features (torch.Tensor): Image features of shape [batch_size, feature_dim].
            text_features (torch.Tensor): Text features of shape [batch_size, feature_dim].
            temperature (float): Temperature parameter for scaling logits.
        Returns:
            torch.Tensor: Computed contrastive loss.
        """
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = image_features @ text_features.T / temperature
        labels = torch.arange(len(image_features)).to(image_features.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            num_epochs: int = 10,
            start_epoch: int = 1
    ) -> list[dict]:
        """
        Train the CLIP model for a specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader providing
                batches of (images, texts).
            num_epochs (int, optional): Number of epochs to train. Defaults
                to 10.
            start_epoch (int, optional): Epoch to start training from (useful
                for resuming). Defaults to 1.

        Returns:
            list[dict]: List of dictionaries containing epoch number and
                average loss per epoch.
        """
        self.model.train()

        if start_epoch == 1:
            total_epochs = num_epochs
        else:
            total_epochs = start_epoch + num_epochs - 1
        for epoch in range(start_epoch, total_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{total_epochs}...")
            total_loss = 0.0
            for images, texts in train_loader:
                images = images.to(self.device)
                texts = texts.to(self.device)

                image_features, text_features, _ = self.model(images, texts)
                loss = self.contrastive_loss(image_features, text_features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.loss_history.append({"epoch": epoch, "loss": avg_loss})
            logger.info(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

        return self.loss_history

    def save_model(self,
                   path: str = "models/clip_finetuned.pt",
                   epoch: int = None) -> None:
        """
        Save the fine-tuned model weights to a file.

        Args:
            path (str): Path to save the model
                (e.g., "models/clip_finetuned.pt").
            epoch (int, optional): Current epoch number for logging.
                Defaults to None.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss_history": self.loss_history if hasattr(self, "loss_history") else None,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")

    @staticmethod
    def load_checkpoint(
            path: str,
            model_name: str = "ViT-B-32",
            lr: float = 1e-5,
            device: str = None
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, int, list]:
        """
        Load a CLIP model and optimizer from a checkpoint.

        Returns:
            model (nn.Module): CLIP model with restored weights.
            optimizer (torch.optim.Optimizer): Restored optimizer.
            start_epoch (int): Last completed epoch.
            loss_history (list): Training loss per epoch.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        try:
            checkpoint = torch.load(path, map_location=device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            raise

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.error(f"Failed to load model or optimizer state: {e}")
            raise

        start_epoch = checkpoint.get("epoch", 0)
        loss_history = checkpoint.get("loss_history", [])

        model.eval()

        return model, optimizer, start_epoch, loss_history
