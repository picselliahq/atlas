from picsellia import DatasetVersion


class DataCardStats:
    def __init__(self, dataset_version: DatasetVersion):
        self.dataset_version = dataset_version
        self.context: dict = {}

    def compute(self) -> "DataCardStats":
        self.context = self.dataset_version.sync()
        return self
