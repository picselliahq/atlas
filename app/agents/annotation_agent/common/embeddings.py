from picsellia import DatasetVersion
from picsellia.types.enums import InferenceType


def load_all_shapes_and_vectors(dataset_version: DatasetVersion, count: int) -> dict:
    vectors = {}
    if dataset_version.type == InferenceType.OBJECT_DETECTION:
        for point in dataset_version.list_rectangles_embeddings(limit=count):
            vectors[point["id"]] = point["vector"]

    shape_ids = list(vectors.keys())
    embeddings_map = {}
    for shape_id in shape_ids:
        try:
            embeddings_map[shape_id] = vectors[shape_id][
                "open_clip||ViT-B-16||datacomp_xl_s13b_b90k"
            ]
        except KeyError:
            pass
    return embeddings_map
