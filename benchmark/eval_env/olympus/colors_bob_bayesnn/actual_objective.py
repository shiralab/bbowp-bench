from olympus import Emulator
from util.base_actual_objective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="colors_bob", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [x["red"], x["orange"], x["yellow"], x["blue"], x["green"]]
        return self.emulator.run(x_input).item()
