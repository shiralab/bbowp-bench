from pydantic import BaseModel

from .algorithm import AlgorithmName
from .search_space import SearchSpace


class SelectedAlgorithm(BaseModel, frozen=True):
    name: AlgorithmName
    reason: str


class DesignedAnswer(BaseModel, frozen=True):
    search_space: SearchSpace
    algorithm: SelectedAlgorithm
