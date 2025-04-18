from agents.common.tools import PicselliaBaseTool


class DeleteImageFromDatasetTool(PicselliaBaseTool):
    name = "delete_image_from_dataset"
    description = """ Use this tool to delete an image in a Picsellia dataset.
        This is useful for removing specific images based on observations or analyses
        in the Picsellia platform.

        If the asset_id is invalid or the image doesn't exist, the code will raise an error. The error will be caught and a specific message will be returned indicating that the asset was not found.

        The format of the returned string is a confirmation message. It indicates the result of the deletion operation, such as whether the asset was successfully removed or if there was an error during the process.
        Deleting an image also removes its associated tags. When an asset is deleted, all metadata including tags associated with that asset are also removed from the dataset.

        usage_instructions:
        - "Call this tool when you need to delete an existing image in a Picsellia dataset."
        - "Do not use this tool for tagging or retrieving images; it is only for deleting an image."
        """
    inputs = {
        "asset_id": {
            "type": "string",
            "description": "id of the asset to delete from the dataset.",
            "nullable": "true",
        },
    }

    output_type = "string"

    def forward(self, asset_id: str | None = None) -> str:
        self.dataset.find_asset(id=asset_id).delete()
        return f"{asset_id} removed from dataset"
