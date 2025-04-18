import picsellia


class AssetMetadata:
    def __init__(self, asset: picsellia.Asset):
        self.asset = asset
        self.id = str(asset.id)
        self.filename = asset.filename
        self.target_path = None
        self.width = asset.width
        self.height = asset.height
        self.url = asset.url
        self.tags = self._get_tags()
        self.clip_embeddings = asset._embeddings

    def _get_tags(self):
        """Fetch the tags associated with the asset."""
        try:
            return [e.name for e in self.asset.data_tags]
        except Exception:
            return []

    def get_metadata(self):
        """Returns general metadata about the asset."""
        return {
            "asset_id": self.id,
            "filename": self.filename,
            "target_path": self.target_path,
            "width": self.width,
            "height": self.height,
            "asset_url": self.url,
            "tags": self.tags,
            "clip_embeddings": self.clip_embeddings,
        }
