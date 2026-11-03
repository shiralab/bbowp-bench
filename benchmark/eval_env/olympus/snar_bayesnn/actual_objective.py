from olympus import Emulator
from util.base_actual_objective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="snar", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [
            x["residence_time"],
            x["ratio"],
            x["concentration"],
            x["temperature"],
        ]
        return self.emulator.run(x_input).item()
