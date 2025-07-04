import logging

import requests
from picsellia import Asset
from PIL import Image

logger = logging.getLogger(__name__)


class AssetImageHandler:
    def __init__(self) -> None:
        pass

    def _get_asset_image(self, asset: Asset) -> Image.Image:
        """Fetch and return the image from the asset's URL."""
        logger.debug(f"Fetching image from asset URL: {asset.url}")
        return Image.open(requests.get(asset.url, stream=True).raw)
