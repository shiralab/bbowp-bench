import logging
import threading

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from bbowp.schema.algorithm import AlgorithmName
from bbowp.schema.bbowp import Direction
from bbowp.schema.designed_answer import DesignedAnswer
from bbowp.schema.execution_result import ExecutionResult
from bbowp.schema.variable_type import ConstVariableType, VariableType
from bbowp.util.objective import Objective
from bbowp.util.sampler import Sampler

_smac_logging_patch_lock = threading.Lock()
_smac_logging_patched = False


def _disable_smac_global_logging() -> None:
    """Prevent SMAC from reconfiguring the process-wide root logger via dictConfig."""
    global _smac_logging_patched
    if _smac_logging_patched:
        return

    with _smac_logging_patch_lock:
        if _smac_logging_patched:
            return

        # SMAC imports setup_logging into abstract_facade, so patch both references.
        import smac.facade.abstract_facade as smac_abstract_facade
        import smac.utils.logging as smac_logging

        def _noop_setup_logging(_level=False):
            return

        smac_logging.setup_logging = _noop_setup_logging
        smac_abstract_facade.setup_logging = _noop_setup_logging
        _smac_logging_patched = True


class Executor:
    """Execute optimization for Objective from DesignedAnswer using Optuna."""

    def __init__(
        self,
        objective: Objective,
        direction: Direction,
        max_eval_value: int,
        designed_answer: DesignedAnswer,
        seed: int,
        logger: logging.Logger,
        preprocess_name_fn=None,
    ):
        """
        Initialize the executor with objective, designed answer, and random seed.
        Args:
            objective (Objective): The objective function or task to be executed.
            direction (Direction): The optimization direction (minimize or maximize).
            max_eval_value (int): The maximum number of evaluations for optimization.
            designed_answer (DesignedAnswer): The designed answer containing the search space and algorithm.
            seed (int): Random seed.
            logger (logging.Logger): Logger for logging optimization process.
            preprocess_name_fn (callable, optional): A function to preprocess variable names for alias mapping. Defaults to None.
        """
        self._objective = objective
        self._direction = direction
        self._max_eval_value = max_eval_value
        self._designed_answer = designed_answer
        self._seed = seed
        self._logger = logger
        self._preprocess_name_fn = None
        if preprocess_name_fn is not None:
            self._preprocess_name_fn = preprocess_name_fn

    def execute(self) -> ExecutionResult:
        """
        Execute optimization.

        Returns:
            ExecutionResult containing best_params and best_fval
        """
        sampler = self._build_sampler()

        study = optuna.create_study(
            direction=self._direction.value,
            sampler=sampler,
        )

        def optuna_objective(trial):
            params = {}

            for var in self._designed_answer.search_space.variables:
                var_name = var.actual_name
                if var_name in params:
                    raise ValueError(
                        f"Duplicate variable name '{var_name}' found in the search space. Variable names must be unique in the search space."
                    )

                if var.type == VariableType.INT:
                    step = var.step
                    if step is None or (var.log and step != 1):
                        step = 1
                    params[var_name] = trial.suggest_int(
                        var_name,
                        var.low,
                        var.high,
                        step=step,
                        log=var.log,
                    )
                elif var.type == VariableType.FLOAT:
                    params[var_name] = trial.suggest_float(
                        var_name,
                        var.low,
                        var.high,
                        log=var.log,
                    )
                elif var.type == VariableType.BOOL:
                    params[var_name] = trial.suggest_categorical(
                        var_name,
                        [True, False],
                    )
                elif var.type == VariableType.CATEGORICAL:
                    choices = [choice.actual_name for choice in var.choices]
                    params[var_name] = trial.suggest_categorical(
                        var_name,
                        choices,
                    )
                elif var.type in [
                    ConstVariableType.INT,
                    ConstVariableType.FLOAT,
                    ConstVariableType.BOOL,
                    ConstVariableType.CATEGORICAL,
                ]:
                    params[var_name] = var.value
                else:
                    raise ValueError(f"Unsupported variable type: {var.type}")

            return self._objective(params, preprocess_name_fn=self._preprocess_name_fn)

        # run optimization
        study.optimize(optuna_objective, n_trials=self._max_eval_value)

        # extract results
        best_params = study.best_trial.params
        for var in self._designed_answer.search_space.variables:
            if (
                var.type
                in [
                    ConstVariableType.INT,
                    ConstVariableType.FLOAT,
                    ConstVariableType.BOOL,
                    ConstVariableType.CATEGORICAL,
                ]
                and var.actual_name not in best_params
            ):
                best_params[var.actual_name] = var.value

        best_fval = study.best_trial.value

        return ExecutionResult(best_params=best_params, best_fval=best_fval)

    def _build_sampler(self) -> optuna.samplers.BaseSampler:
        optuna_search_space = self._define_search_space()
        algorithm = self._designed_answer.algorithm.name

        if algorithm == AlgorithmName.SMAC3:
            _disable_smac_global_logging()

        return Sampler.get_sampler(
            algorithm,
            seed=self._seed,
            search_space=optuna_search_space,
            n_trials=self._max_eval_value,
        )

    def _define_search_space(self):
        search_space = {}
        for var in self._designed_answer.search_space.variables:
            if var.type == VariableType.INT:
                step = var.step
                if step is None:
                    step = 1
                elif var.log and step != 1:
                    self._logger.warning(
                        f"Int variable '{var.actual_name}' has log=True but step={step}. Optuna does not support step > 1 for log-scaled integer variables. Overriding step to 1."
                    )
                    step = 1
                search_space[var.actual_name] = IntDistribution(
                    low=var.low,
                    high=var.high,
                    step=step,
                    log=var.log,
                )
            elif var.type == VariableType.FLOAT:
                search_space[var.actual_name] = FloatDistribution(
                    low=var.low,
                    high=var.high,
                    log=var.log,
                )
            elif var.type == VariableType.BOOL:
                search_space[var.actual_name] = CategoricalDistribution(
                    choices=[True, False]
                )
            elif var.type == VariableType.CATEGORICAL:
                search_space[var.actual_name] = CategoricalDistribution(
                    choices=[choice.actual_name for choice in var.choices]
                )
            elif var.type in [
                ConstVariableType.INT,
                ConstVariableType.FLOAT,
                ConstVariableType.BOOL,
                ConstVariableType.CATEGORICAL,
            ]:
                # Constant variables are not included in the search space since they are not optimized.
                continue
            else:
                raise ValueError(f"Unsupported variable type: {var.type}")

        return search_space
