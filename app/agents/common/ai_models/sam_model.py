import torch
from PIL import Image
from ultralytics import FastSAM

from config.settings import settings


class SAMModel:
    def __init__(self) -> None:
        self.model = FastSAM(settings.fast_sam_path)
        self.device = (
            "mps"
            if torch.mps.is_available()
            else ("0" if torch.cuda.is_available() else "cpu")
        )

    def predict(self, image: Image, box_prompt: list, label: str):
        """Uses the SAM model to predict boxes from an image."""
        results = self.model(
            image,
            labels=label,
            bboxes=box_prompt,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def post_process(self, results, image_size: tuple):
        """Post-process the results to get final box coordinates."""
        sam_boxes = []
        img_w, img_h = image_size

        for result in results:
            x, y, w, h = result.boxes.xywhn.cpu().numpy()[0]
            # Adjust the box coordinates according to the original image size
            adjusted_x = (x - w / 2) * img_w
            adjusted_y = (y - h / 2) * img_h
            adjusted_w = w * img_w
            adjusted_h = h * img_h
            sam_boxes.append(
                (int(adjusted_x), int(adjusted_y), int(adjusted_w), int(adjusted_h))
            )

        return sam_boxes
