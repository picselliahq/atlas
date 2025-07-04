from pydantic import BaseModel


class DataCard(BaseModel):
    creator: str
    name: str
    version: str
    dataset_size: int
    task: str
    description: str
    verbose_description: str
