from olympus import Emulator
from util.BaseActualObjective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="colors_n9", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [x["red"], x["green"], x["blue"]]
        return self.emulator.run(x_input).item()
