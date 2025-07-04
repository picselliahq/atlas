import gc

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class CaptioningModel:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate a caption for a given image using the BLIP model.
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        del inputs
        del out
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return caption
