from enum import Enum


class PossibleActions(str, Enum):
    DELETE = "DELETE"
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    TAG = "TAG"
    CLEAN = "CLEAN"
