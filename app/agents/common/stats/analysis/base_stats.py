from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd

SelfStats = TypeVar("SelfStats", bound="BaseStats")


class BaseStats(ABC, Generic[SelfStats]):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def compute(self) -> SelfStats: ...
