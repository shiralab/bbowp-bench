from olympus import Emulator
from util.base_actual_objective import BaseActualObjective


class ActualObjective(BaseActualObjective):
    def __init__(self):
        super().__init__()
        self.emulator = Emulator(dataset="photo_wf3", model="BayesNeuralNet")

    def __call__(self, x) -> float:
        x_input = [x["mat_1"], x["mat_2"], x["mat_3"], x["mat_4"]]
        return self.emulator.run(x_input).item()
