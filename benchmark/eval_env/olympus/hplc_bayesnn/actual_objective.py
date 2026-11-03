from olympus import Emulator
from util.BaseActualObjective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="hplc", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [
            x["sample_loop"],
            x["additional_volume"],
            x["tubing_volume"],
            x["sample_flow"],
            x["push_speed"],
            x["wait_time"],
        ]
        return self.emulator.run(x_input).item()
