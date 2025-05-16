from PIL import Image

from services.embedders import clip_shape_embedder


class LocalAnnotationMetadata:
    def __init__(self, coco_item: dict, image: Image):
        self.filename = coco_item["filename"]
        self.image_width = coco_item["image_width"]
        self.image_height = coco_item["image_height"]
        self.shape_id = coco_item["shape_id"]

        self.label = coco_item["label"]
        self.label_id = coco_item["label_id"]
        self.x = coco_item["x"]
        self.y = coco_item["y"]
        self.w = coco_item["w"]
        self.h = coco_item["h"]
        self.x_norm = self.x / self.image_width
        self.y_norm = self.y / self.image_height
        self.w_norm = self.w / self.image_width
        self.h_norm = self.h / self.image_height
        x2 = self.x + self.w
        y2 = self.y + self.h
        self.shape_embeddings = clip_shape_embedder.extract_rectangle_embedding(image, [(self.x, self.y), (x2, y2)])

    def dict(self):
        """Return a dictionary representation of the metadata."""
        return {
            "filename": self.filename,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "shape_id": self.shape_id,
            "label": self.label,
            "label_id": self.label_id,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "w_norm": self.w_norm,
            "h_norm": self.h_norm,
            "shape_embeddings": self.shape_embeddings.tolist(),
        }
