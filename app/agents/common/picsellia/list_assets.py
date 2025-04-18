import math
from uuid import UUID

from picsellia import DatasetVersion, Tag
from picsellia.sdk.asset import Asset, MultiAsset


def list_paginated_ids(
    dataset_version: DatasetVersion, offset: int, limit: int
) -> tuple[list[UUID], int]:
    r = dataset_version.connexion.get(
        f"/api/dataset/version/{dataset_version.id}/assets/ids/paginated",
        params={"limit": 200, "offset": offset},
    ).json()
    return r["items"], r["count"]


def list_preview_assets_from_ids(
    dataset_version: DatasetVersion, ids: list[UUID]
) -> list[Asset]:
    r = dataset_version.connexion.get(
        f"/api/dataset/version/{dataset_version.id}/assets/preview",
        params={"ids": ids},
    ).json()
    assets = []
    for _, item in r["items"].items():
        asset = Asset(dataset_version.connexion, dataset_version.id, item)
        asset.annotation_id = item["annotation"]["id"] if item["annotation"] else None
        asset.tags = [Tag(dataset_version.connexion, tag) for tag in item["tags"]]
        asset.data_tags = [
            Tag(dataset_version.connexion, tag) for tag in item["data"]["tags"]
        ]
        assets.append(asset)
    return assets


def load_assets(dataset_version: DatasetVersion) -> MultiAsset:
    items = []
    offset = 0
    count = 1
    limit = 200
    ids = []
    while offset < count:
        page_ids, count = list_paginated_ids(dataset_version, offset, limit)
        ids.extend(page_ids)
        offset += limit

    batch_size = 50
    batch_count = math.ceil(len(ids) / batch_size)
    for batch in range(batch_count):
        batch_ids = ids[batch_size * batch : batch_size * (batch + 1)]
        items.extend(list_preview_assets_from_ids(dataset_version, batch_ids))

    return MultiAsset(dataset_version.connexion, dataset_version.id, items)
