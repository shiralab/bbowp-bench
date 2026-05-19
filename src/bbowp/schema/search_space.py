import abc
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .variable_type import ConstVariableType, VariableType


class AbstractDesignVariable(BaseModel, abc.ABC, frozen=True):
    actual_name: str  # The argument name used in the actual objective function


class CategoryChoice(BaseModel, frozen=True):
    actual_name: (
        str  # The choice name of category used in the actual objective function
    )


class IntDesignVariable(AbstractDesignVariable, frozen=True):
    low: int
    high: int
    step: int = 1
    log: bool = False

    type: Literal[VariableType.INT] = VariableType.INT

    def model_post_init(self, context: object) -> None:
        if self.step <= 0:
            raise ValueError(f"Step must be positive, got {self.step}")
        if self.low >= self.high:
            raise ValueError(
                f"Low must be less than high, got low={self.low}, high={self.high}"
            )


class FloatDesignVariable(AbstractDesignVariable, frozen=True):
    low: float
    high: float
    log: bool = False

    type: Literal[VariableType.FLOAT] = VariableType.FLOAT

    def model_post_init(self, context: object) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Low must be less than high, got low={self.low}, high={self.high}"
            )


class BoolDesignVariable(AbstractDesignVariable, frozen=True):
    type: Literal[VariableType.BOOL] = VariableType.BOOL


class CategoricalDesignVariable(AbstractDesignVariable, frozen=True):
    choices: tuple[CategoryChoice, ...]

    type: Literal[VariableType.CATEGORICAL] = VariableType.CATEGORICAL

    def model_post_init(self, context: object) -> None:
        if len(self.choices) == 0:
            raise ValueError(
                f"Variable '{self.actual_name}' must have non-empty choices"
            )


class ConstIntVariable(AbstractDesignVariable, frozen=True):
    value: int

    type: Literal[ConstVariableType.INT] = ConstVariableType.INT


class ConstFloatVariable(AbstractDesignVariable, frozen=True):
    value: float

    type: Literal[ConstVariableType.FLOAT] = ConstVariableType.FLOAT


class ConstBoolVariable(AbstractDesignVariable, frozen=True):
    value: bool

    type: Literal[ConstVariableType.BOOL] = ConstVariableType.BOOL


class ConstCategoricalVariable(AbstractDesignVariable, frozen=True):
    value: str

    type: Literal[ConstVariableType.CATEGORICAL] = ConstVariableType.CATEGORICAL


DesignVariable = Annotated[
    IntDesignVariable
    | FloatDesignVariable
    | BoolDesignVariable
    | CategoricalDesignVariable
    | ConstIntVariable
    | ConstFloatVariable
    | ConstBoolVariable
    | ConstCategoricalVariable,
    Field(discriminator="type"),
]


class SearchSpace(BaseModel, frozen=True):
    variables: tuple[DesignVariable, ...]
    reason: str
