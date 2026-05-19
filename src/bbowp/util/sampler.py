import optuna
import optunahub
from optuna.distributions import BaseDistribution

from bbowp.schema.algorithm import AlgorithmName


class Sampler:
    """Factory for creating Optuna samplers based on algorithm name."""

    @staticmethod
    def get_sampler(
        algorithm: AlgorithmName,
        seed: int = 42,
        search_space: dict[str, BaseDistribution] | None = None,
        n_trials: int | None = None,
    ) -> optuna.samplers.BaseSampler:
        """
        Create an Optuna sampler based on the algorithm name.

        Args:
            algorithm: The algorithm to use (AlgorithmName enum)
            seed: Random seed for reproducibility
            search_space: Optional search space definition, required to initialize some samplers
            n_trials: Optional number of trials, required to initialize some samplers

        Returns:
            An Optuna sampler instance

        Raises:
            NotImplementedError: If the algorithm is not yet supported
        """
        if algorithm == AlgorithmName.RANDOM:
            return optuna.samplers.RandomSampler(seed=seed)
        if algorithm == AlgorithmName.GP:
            return optuna.samplers.GPSampler(seed=seed)
        if algorithm == AlgorithmName.TPE:
            return optuna.samplers.TPESampler(seed=seed)
        if algorithm == AlgorithmName.CMA_ES:
            return optuna.samplers.CmaEsSampler(seed=seed)
        if algorithm == AlgorithmName.CMA_ES_WM:
            return optuna.samplers.CmaEsSampler(seed=seed, with_margin=True)
        if algorithm == AlgorithmName.CATCMA:
            catcma_module = optunahub.load_module("samplers/catcma")
            return catcma_module.CatCmaSampler(seed=seed)
        if algorithm == AlgorithmName.CATCMA_WM:
            catcmawm_module = optunahub.load_module("samplers/catcmawm")
            return catcmawm_module.CatCmawmSampler(seed=seed)

        if algorithm == AlgorithmName.SMAC3:
            if search_space is None:
                raise ValueError("search_space is required for SMAC3 sampler")
            if n_trials is None:
                raise ValueError("n_trials is required for SMAC3 sampler")
            smac_module = optunahub.load_module("samplers/smac_sampler")
            return smac_module.SMACSampler(search_space, n_trials=n_trials, seed=seed)

        if algorithm == AlgorithmName.NELDER_MEAD:
            if search_space is None:
                raise ValueError("search_space is required for Nelder-Mead sampler")
            nelder_mead_module = optunahub.load_module("samplers/nelder_mead")
            return nelder_mead_module.NelderMeadSampler(search_space, seed=seed)

        if algorithm == AlgorithmName.PSO:
            if search_space is None:
                raise ValueError("search_space is required for PSO sampler")
            pso_module = optunahub.load_module("samplers/pso")
            return pso_module.PSOSampler(search_space=search_space, seed=seed)

        if algorithm == AlgorithmName.DE:
            de_module = optunahub.load_module("samplers/differential_evolution")
            return de_module.DESampler(seed=seed)

        raise NotImplementedError(f"Algorithm '{algorithm}' is not yet supported.")
