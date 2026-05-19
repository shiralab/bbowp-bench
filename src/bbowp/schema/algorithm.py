from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class AlgorithmName(StrEnum):
    RANDOM = "Random Search"  # for baseline
    GP = "GPSampler"
    TPE = "TPE"
    SMAC3 = "SMAC3"
    CMA_ES = "CMA-ES"
    CMA_ES_WM = "CMA-ES with margin"
    CATCMA = "CatCMA"
    CATCMA_WM = "CatCMA with margin"
    DE = "Differential Evolution"
    NELDER_MEAD = "Nelder-Mead"
    PSO = "Particle Swarm Optimization"


class Algorithm(BaseModel, frozen=True):
    name: AlgorithmName
    reason: str
    score: float | Literal["TBA"]
