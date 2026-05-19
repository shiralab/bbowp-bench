from enum import Enum

from pydantic import BaseModel

from .algorithm import Algorithm
from .search_space import SearchSpace
from .variable_type import VariableType


class Difficulty(str, Enum):
    """Define difficulty levels.

    EASY: Describes tasks with explicit parameters to be optimized and their ranges.
    INTERMEDIATE: Describes tasks with explicit parameters to be optimized, but without specified ranges.
    HARD: Describes tasks without explicit parameters to be optimized.
    """

    EASY = "easy"
    INTERMEDIATE = "intermediate"
    HARD = "hard"


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ProblemMetadata(BaseModel, frozen=True):
    domain: str
    source: str


class Choice(BaseModel, frozen=True):
    actual_name: str
    aliases: tuple[str, ...]


class ObjectiveArgument(BaseModel, frozen=True):
    actual_name: str
    aliases: tuple[str, ...]
    type: VariableType
    low: int | float | None = None
    high: int | float | None = None
    choices: tuple[Choice, ...] | None = None
    default: int | float | bool | str

    def model_post_init(self, context: object) -> None:
        if self.type in (VariableType.FLOAT, VariableType.INT):
            if self.choices is not None:
                raise ValueError(
                    f"Numeric variable '{self.actual_name}' must have choices=null, got: {self.choices}"
                )
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Numeric variable '{self.actual_name}' must have low and high defined, got: low={self.low}, high={self.high}"
                )

            if not (self.low <= self.default <= self.high):
                raise ValueError(
                    f"Default value of variable '{self.actual_name}' must be within the range defined by low and high, got: default={self.default}, low={self.low}, high={self.high}"
                )
        elif self.type == VariableType.INT:
            if not isinstance(self.default, int):
                raise ValueError(
                    f"Default value of integer variable '{self.actual_name}' must be an integer, got: {self.default}"
                )
        elif self.type == VariableType.FLOAT:
            if not isinstance(self.default, (int, float)):
                raise ValueError(
                    f"Default value of float variable '{self.actual_name}' must be a number, got: {self.default}"
                )

        elif self.type == VariableType.CATEGORICAL:
            if self.choices is None or len(self.choices) == 0:
                raise ValueError(
                    f"Categorical variable '{self.actual_name}' must have non-empty choices"
                )
            if self.low is not None or self.high is not None:
                raise ValueError(
                    f"Categorical variable '{self.actual_name}' must have low=null and high=null, got: low={self.low}, high={self.high}"
                )

        elif self.type == VariableType.BOOL:
            if self.choices is not None and len(self.choices) > 0:
                raise ValueError(
                    f"Boolean variable '{self.actual_name}' must have choices=null, got: {self.choices}"
                )
            if self.low is not None or self.high is not None:
                raise ValueError(
                    f"Boolean variable '{self.actual_name}' must have low=null and high=null, got: low={self.low}, high={self.high}"
                )
            if not isinstance(self.default, bool):
                raise ValueError(
                    f"Boolean variable '{self.actual_name}' must have default value of True or False, got: {self.default}"
                )
        else:
            raise ValueError(
                f"Unsupported variable type '{self.type}' for variable '{self.actual_name}'"
            )


class ExecutionConfig(BaseModel, frozen=True):
    docker_image: str
    arguments: tuple[ObjectiveArgument, ...]


class OptimizationTarget(BaseModel, frozen=True):
    metric: str
    direction: Direction


class ExpertAnswerItem(BaseModel, frozen=True):
    search_space: SearchSpace
    algorithm: Algorithm


class ExpertAnswer(BaseModel, frozen=True):
    amount: str
    answer_item: ExpertAnswerItem


class Budget(BaseModel, frozen=True):
    amount: str
    max_eval_value: int


class Description(BaseModel, frozen=True):
    difficulty: Difficulty
    text: str


class BBOWPEvalConfig(BaseModel, frozen=True):
    problem_id: str
    target: OptimizationTarget
    execution: ExecutionConfig


class BBOWP(BaseModel, frozen=True):
    problem_id: str
    meta_data: ProblemMetadata
    budget: tuple[Budget, ...]
    description: tuple[Description, ...]
    answer: tuple[ExpertAnswer, ...]
