from olympus import Emulator
from util.base_actual_objective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="suzuki", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [x["temperature"], x["pd_mol"], x["arbpin"], x["k3po4"]]
        return self.emulator.run(x_input).item()
