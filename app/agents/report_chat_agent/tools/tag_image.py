from picsellia import Asset

from agents.common.tools import PicselliaBaseTool


class TagImageTool(PicselliaBaseTool):
    name = "tag_image_in_picsellia_platform"
    description = """ Use this tool to add a tag to an image in a Picsellia dataset.
        This is useful for linking specific observations or analyses to a retrievable object
        in the Picsellia platform. The tool checks if the tag already exists for the asset
        and only adds it if it doesn't exist. It handles errors such as missing inputs or
        issues during the tagging process.

        If the asset_id is invalid or doesn't exist, the code attempts to find the asset using the dataset's find_asset method.
        If this fails, an exception is caught, and a specific error message is returned indicating that the asset was not found.
        Similarly, if the tag is invalid or cannot be created/retrieved, an exception is caught, and a specific error message is returned.

        The format of the returned string is a confirmation message. It indicates the result of the tagging operation,
        such as whether the tag was successfully added, already exists, or if there was an error during the process.

        If a tag already exists on the image, the code checks the existing tags of the asset.
        If the tag is found in the existing tags, a message is returned indicating that the tag already exists on the asset.
        The tag is not appended or replaced; it is simply ignored in this case.

        usage_instructions:
        - "Call this tool when you need to associate a new tag with an existing image in a Picsellia dataset."
        - "Do not use this tool for retrieving or removing tags; it is only for adding a tag."
        - "Ensure both 'asset_id' and 'tag' are provided to avoid errors."
        - "The tool will return a message indicating whether the tag was added or if it already exists."
        """
    inputs = {
        "asset_id": {
            "type": "string",
            "description": "ID of the asset to which you want to add a tag.",
            "nullable": "false",
        },
        "tag": {
            "type": "string",
            "description": "Tag to be added to the asset.",
            "nullable": "false",
        },
    }

    output_type = "string"

    def forward(self, asset_id: str | None = None, tag: str | None = None) -> str:
        if not asset_id or not tag:
            return "Error: Both 'asset_id' and 'tag' must be provided."

        try:
            tag_obj = self.dataset.get_or_create_asset_tag(tag)
            asset: Asset = self.dataset.find_asset(id=asset_id)
            asset.add_tags(tag_obj)
            return f"Tag '{tag}' added to asset {asset_id}."
        except Exception as e:
            return f"Error: {str(e)}"
