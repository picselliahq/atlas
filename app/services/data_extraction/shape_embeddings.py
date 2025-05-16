from PIL import Image
import numpy as np
from typing import Union, List, Tuple
from services.data_extraction.image_embeddings import CLIPImageEmbedder


class CLIPShapeEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the shape embedder with a CLIP model.

        Args:
            model_name (str): The name of the CLIP model to load. Defaults to "openai/clip-vit-base-patch32"
        """
        self.embedder = CLIPImageEmbedder(model_name)

    @staticmethod
    def _get_bounding_box(
            shape: List[Tuple[float, float]], is_polygon: bool
    ) -> Tuple[int, int, int, int]:
        """
        Helper function to compute bounding box coordinates for a shape.

        Args:
            shape (List[Tuple[float, float]]): List of points defining the shape
            is_polygon (bool): Whether the shape is a polygon (True) or rectangle (False)

        Returns:
            Tuple[int, int, int, int]: (left, top, right, bottom) coordinates
        """
        if is_polygon:
            x_coords = [p[0] for p in shape]
            y_coords = [p[1] for p in shape]
            left = min(x_coords)
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)
        else:
            left = min(shape[0][0], shape[1][0])
            top = min(shape[0][1], shape[1][1])
            right = max(shape[0][0], shape[1][0])
            bottom = max(shape[0][1], shape[1][1])

        return int(left), int(top), int(right), int(bottom)

    def extract_shape_embedding(
            self,
            image: Image.Image,
            shapes: Union[
                List[Tuple[float, float]],  # Single rectangle: [(x1,y1), (x2,y2)]
                List[List[Tuple[float, float]]],  # Multiple rectangles or single polygon
            ],
            is_polygon: bool = True,
    ) -> np.ndarray:
        """
        Extract CLIP embeddings for one or more shapes in an image.

        Args:
            image (Image.Image): The input image
            shapes: Can be one of:
                - List of 2 points [(x1,y1), (x2,y2)] for a single rectangle
                - List of points [(x1,y1), (x2,y2), ...] for a single polygon
                - List of the above for multiple shapes
            is_polygon (bool): Whether the shapes are polygons (True) or rectangles (False)

        Returns:
            np.ndarray: Array of embeddings with shape (n_shapes, embedding_dim)
        """
        # Convert single shape to list
        if not isinstance(shapes[0], list):
            shapes = [shapes]

        cropped_images = []

        for shape in shapes:
            left, top, right, bottom = self._get_bounding_box(shape, is_polygon)

            # Ensure coordinates are within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(image.width, right)
            bottom = min(image.height, bottom)

            # Crop the image
            cropped = image.crop((left, top, right, bottom))
            cropped_images.append(cropped)

        # Compute embeddings for all cropped images
        return self.embedder.compute_embeddings(cropped_images)

    def extract_rectangle_embedding(
            self,
            image: Image.Image,
            rectangles: Union[
                List[Tuple[float, float]],  # Single rectangle: [(x1,y1), (x2,y2)]
                List[List[Tuple[float, float]]],  # Multiple rectangles
            ],
    ) -> np.ndarray:
        """
        Convenience function to extract embeddings for rectangles.
        """
        return self.extract_shape_embedding(image, rectangles, is_polygon=False)

    def extract_polygon_embedding(
            self,
            image: Image.Image,
            polygons: Union[
                List[Tuple[float, float]],  # Single polygon: [(x1,y1), (x2,y2), ...]
                List[List[Tuple[float, float]]],  # Multiple polygons
            ],
    ) -> np.ndarray:
        """
        Convenience function to extract embeddings for polygons.
        """
        return self.extract_shape_embedding(image, polygons, is_polygon=True)
