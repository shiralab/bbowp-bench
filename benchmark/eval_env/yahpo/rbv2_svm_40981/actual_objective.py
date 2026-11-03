from util.base_actual_objective import BaseActualObjective
from yahpo_gym import benchmark_set


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.bench = benchmark_set.BenchmarkSet("rbv2_svm")
        self.bench.set_instance("40981")
        self.cs = self.bench.get_opt_space()

    def __call__(self, x) -> float:
        fixed = {
            "task_id": "40981",
            "repl": 10,
            "trainsize": 1.0,
        }
        x.update(fixed)
        return self.bench.objective_function(self.get_valid_x(x))[0]["auc"]
