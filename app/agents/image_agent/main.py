import traceback

import pandas as pd

from agents.common.tools import PicselliaBaseTool
from agents.image_agent.common.get_context import PContext
from agents.image_agent.tools import ALL_TOOLS


def run_analysis(pctx: PContext) -> PContext:
    for tool in ALL_TOOLS:
        try:
            tool(pctx)
        except Exception:
            print(f"❌ Tool `{tool.__name__}` failed:")
            print(traceback.format_exc())
    pctx.sync()
    return pctx


class ImageAnalyticsTool(PicselliaBaseTool):
    name = "retrieve_dataset_analysis_dictionnary_tool"
    description = """
        Use this tool to retrieve a detailed analysis dictionnary for a Picsellia dataset.
        The dictionnary includes information about outlier groups, contrast analysis, luminance analysis,
        semantic analysis, and quality analysis. It is helpful when you need to interpret
        or act upon specific findings such as tagging or deleting images based on identified conditions.

        usage_instructions:
        - "Call this tool when you need to fetch the entire analysis data structure to review outlier images,
            examine contrast or luminance distributions, or observe semantic groupings."
        - "Use it prior to any logic that requires an overview of the dataset’s image issues (e.g., blurry or low-contrast images)
            or grouping insights (e.g., ‘Wind Turbines Near Water’, etc.)."
        - "Do not use this tool if you only need to apply a single tag to one image (use your dedicated TagImageTool instead)."
        - "Do not call this tool repeatedly if you already have the analysis JSON available."

        usage_constraints:
        - "Ensure the `dataset_id` is valid and accessible through the Picsellia platform."
        - "The response structure will always follow the schema outlined above."
        - "Use caution when parsing large JSON responses; always check for the presence of the required keys
            (`outlier_groups`, `contrast_analysis`, `luminance_analysis`, `semantic_analysis`, and `quality_analysis`)."

        examples:
        - description: "Retrieve the analysis JSON for dataset with ID 'str'."
            input:
            dataset_id: "str"
            output:
            outlier_groups:
                - group_type: "OUTLIER"
                images:
                    - asset_id: "01955cdb-3606-7772-af85-8caccabf2725"
                    potential_actions: ["TAG", "DELETE"]
                    - asset_id: "01955cdb-3545-77e9-bcb3-4c3c5b04cb13"
                    potential_actions: ["TAG", "DELETE"]
                    # ...additional items...
            contrast_analysis:
                - executive_summary: "The contrast distribution graph provides..."
                insights: "The coherence of contrast values across this dataset..."
                value_analyzed: "IMAGE CONTRAST"
                outlier_images: []
            luminance_analysis:
                - executive_summary: "The luminance distribution graph presented implies..."
                insights: "The luminance graph shows a distribution that..."
                value_analyzed: "IMAGE LUMINANCE"
                outlier_images: []
            semantic_analysis:
                - group_name: "Wind Turbine Fields"
                insights: "This group likely represents images of wind turbines..."
                human_sounding_caption: "Images in this group showcase wind turbines..."
                named_entities: ["wind turbines", "fields", "countryside", "sky"]
                group_id: 0
                images:
                    - asset_id: "01955cdb-3606-7a74-a768-69034ac28ae5"
                    potential_actions: ["TAG"]
                    # ...additional items...
                # ...additional groups...
            quality_analysis:
                - asset_id: "01955cdb-3603-7b93-8cf4-ede06db5e554"
                issues: ["BLURRY"]
                potential_actions: ["DELETE"]

        WHEN TO USE THE JSON ANALYSIS RETRIEVAL TOOL:
        1. Pre-processed Insights:
        - When you need quick access to already-computed analyses
        - When you want standardized insights without writing custom code
        - When you need consistent formatting for reports or dashboards

        2. Semantic Analysis:
        - When you need insights on semantic groups already identified in the dataset
        - When you want pre-computed named entities and their relationships
        - When you need human-readable captions for semantic clusters

        3. Quality Assessment:
        - When you need standardized quality metrics like luminance and contrast analysis
        - When you want executive summaries about dataset quality
        - When you need coherence analysis for specific visual properties

        4. Decision Making:
        - When you need recommendations on which images to keep, review or delete
        - When you want to identify out-of-scope content
        - When you need quick insights to guide dataset curation decisions

        WHEN TO USE THE GET DATASET DATAFRAME TOOL:
        1. Direct Data Analysis:
        - When you need to perform custom calculations on the raw data
        - When you want to explore statistical properties not covered by pre-built analyses
        - When you need to create custom visualizations from the original data points

        2. Data Transformation:
        - When you need to filter, group, or transform the dataset in ways specific to your analysis
        - When you want to apply custom preprocessing before analysis
        - When you need to extract specific subsets of the data based on complex criteria

        3. Advanced Analysis:
        - When you need to perform correlation analysis between multiple columns
        - When you want to run your own clustering or dimensionality reduction methods
        - When you need to test different quality thresholds or metrics

        4. Integration:
        - When you're building a pipeline that requires the raw data
        - When you need to combine this dataset with other data sources
        - When you want to export the data in a different format
        """
    inputs = {
        "dataset_id": {
            "type": "string",
            "description": "ID of the dataset that you want to retrieve the information from. If you don't know this information, leave it blank",
            "nullable": "true",
        }
    }
    output_type = "object"

    def forward(self, dataset_id: str | None = None) -> pd.DataFrame:
        pctx = PContext(context_service=self.context_service)
        pctx = run_analysis(pctx)
        return pctx.df


class GetDatasetDataframe(PicselliaBaseTool):
    name = "get_dataset_dataframe"
    description = """
    DESCRIPTION:
        This tool returns the raw pandas DataFrame containing comprehensive metadata for a computer vision dataset.
    OUTPUT:
    A pandas DataFrame with the following columns:
    - filename: The name of the image file
    - asset_id: Unique identifier for each image asset
    - clip_embeddings: Vector embeddings from CLIP model
    - asset_url: URL to access the image
    - tags: Tags associated with the image
    - caption: Image description/caption
    - color: Color information of the image
    - luminance: Brightness distribution values
    - contrast: Contrast measurement values
    - is_blurry: Boolean indicating if image is detected as blurry
    - is_corrupted: Boolean indicating if image has corruption
    - blur_score: Numerical score representing blur level
    - width: Image width in pixels
    - height: Image height in pixels
    - file_size_bytes: File size in bytes
    - pca_x, pca_y: PCA dimensionality reduction coordinates
    - pca_cluster: Cluster assignment from PCA
    - tsne_x, tsne_y: t-SNE dimensionality reduction coordinates
    - tsne_cluster: Cluster assignment from t-SNE
    - x_umap, y_umap: UMAP dimensionality reduction coordinates
    - umap_cluster: Cluster assignment from UMAP

    WHEN TO USE THE GET DATASET DATAFRAME TOOL:
        1. Direct Data Analysis:
        - When you need to perform custom calculations on the raw data
        - When you want to explore statistical properties not covered by pre-built analyses
        - When you need to create custom visualizations from the original data points

        2. Data Transformation:
        - When you need to filter, group, or transform the dataset in ways specific to your analysis
        - When you want to apply custom preprocessing before analysis
        - When you need to extract specific subsets of the data based on complex criteria

        3. Advanced Analysis:
        - When you need to perform correlation analysis between multiple columns
        - When you want to run your own clustering or dimensionality reduction methods
        - When you need to test different quality thresholds or metrics

        4. Integration:
        - When you're building a pipeline that requires the raw data
        - When you need to combine this dataset with other data sources
        - When you want to export the data in a different format

    WHEN TO USE THE JSON ANALYSIS RETRIEVAL TOOL:
        1. Pre-processed Insights:
        - When you need quick access to already-computed analyses
        - When you want standardized insights without writing custom code
        - When you need consistent formatting for reports or dashboards

        2. Semantic Analysis:
        - When you need insights on semantic groups already identified in the dataset
        - When you want pre-computed named entities and their relationships
        - When you need human-readable captions for semantic clusters

        3. Quality Assessment:
        - When you need standardized quality metrics like luminance and contrast analysis
        - When you want executive summaries about dataset quality
        - When you need coherence analysis for specific visual properties

        4. Decision Making:
        - When you need recommendations on which images to keep, review or delete
        - When you want to identify out-of-scope content
        - When you need quick insights to guide dataset curation decisions

        COMBINING BOTH TOOLS:
        For comprehensive analysis, you might use both tools in sequence:
        1. Start with the JSON analysis tool to get high-level insights and identify areas of interest
        2. Then use the DataFrame tool to dive deeper into specific aspects that require custom analysis
        3. Validate findings from pre-computed analyses with your own calculations on the raw data
    """

    inputs = {
        "dataset_id": {
            "type": "string",
            "description": "ID of the dataset that you want to retrieve the information from. If you don't know this information, leave it blank",
            "nullable": "true",
        }
    }
    output_type = "object"

    def forward(self, dataset_id: str | None = None) -> pd.DataFrame:
        if dataset_id is None:
            dataset_id = str(self.dataset.id)
        pctx = PContext(dataset_id=dataset_id)
        return pctx.df
