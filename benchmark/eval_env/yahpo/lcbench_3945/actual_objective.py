from util.BaseActualObjective import BaseActualObjective
from yahpo_gym import benchmark_set


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.bench = benchmark_set.BenchmarkSet("lcbench")
        self.bench.set_instance("3945")
        self.cs = self.bench.get_opt_space()

    def __call__(self, x) -> float:
        fixed = {"OpenML_task_id": "3945", "epoch": 52}
        x.update(fixed)
        return self.bench.objective_function(self.get_valid_x(x))[0]["val_accuracy"]
