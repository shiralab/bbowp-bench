class BaseActualObjective:
    def __init__(self):
        self.emulator = None

    def __call__(self, x) -> float:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
