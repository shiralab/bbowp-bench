from olympus import Emulator
from util.BaseActualObjective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="fullerenes", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [x["reaction_time"], x["sultine"], x["temperature"]]
        return self.emulator.run(x_input).item()
