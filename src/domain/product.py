from pathlib import Path


class Product:
    """ Class representing a fashion product and its attributes."""

    def __init__(self, product_id: int, name: str, description: str, group: str, color: str, image: Path, master_category: str):
        """ Initializes a Product instance with the given attributes.

        Args:
            product_id: Unique identifier for the product.
            name: Name of the product.
            description: Detailed description of the product.
            group: Category or group to which the product belongs.
            color: Color of the product.
            image: Path to the product image file.
            master_category: Master category of the product.
        """
        self.id = product_id
        self.name = name
        self.description = description
        self.group = group
        self.color = color
        self.image = image
        self.master_category = master_category

    def to_json(self):
        """Converts the Product instance to a JSON serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "color": self.color,
            "image": str(self.image),
            "master_category": self.master_category
        }

    def __repr__(self):
        """String representation of the Product instance."""
        return (f"Product(id={self.id}, name={self.name}, description={self.description}, group={self.group}, "
                f"color={self.color}, image={self.image}, master_category={self.master_category})")
