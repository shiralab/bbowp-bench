from olympus import Emulator
from util.BaseActualObjective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="benzylation", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [x["flow_rate"], x["ratio"], x["solvent"], x["temperature"]]
        return self.emulator.run(x_input).item()
