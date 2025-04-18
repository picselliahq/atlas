from picsellia import DatasetVersion
from picsellia.sdk.asset import MultiAsset


def load_all_assets_and_vectors(
    assets: MultiAsset, dataset_version: DatasetVersion, count: int
) -> MultiAsset:
    vectors = {}
    for point in dataset_version.list_embeddings(limit=count):
        vectors[point["id"]] = point["vector"]

    for asset in assets:
        asset._embeddings = vectors[str(asset.data_id)][
            "open_clip||ViT-B-16||datacomp_xl_s13b_b90k"
        ]
    return assets
