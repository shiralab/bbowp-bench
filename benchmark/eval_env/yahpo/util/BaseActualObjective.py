from yahpo_gym import local_config


class BaseActualObjective:
    def __init__(self):
        local_config.init_config()
        local_config.set_data_path("./config_data")
        self.bench = None
        self.cs = None

    def get_valid_x(self, x: dict) -> dict:
        valid_x = {}
        for param in self.cs.get_hyperparameters():
            name = param.name
            conditions = self.cs.get_parent_conditions_of(name)
            is_active = True
            # resolve conditions
            for cond in conditions:
                required_parents = cond.get_parents()
                parents_are_active = True
                for p in required_parents:
                    if p.name not in valid_x:
                        parents_are_active = False
                        break
                if not parents_are_active:
                    is_active = False
                    break
                if not cond.evaluate(valid_x):
                    is_active = False
                    break
            if is_active:
                if x[name] is not None:
                    valid_x[name] = x[name]
                    # clip values to valid ranges
                    if hasattr(param, "lower"):
                        valid_x[name] = max(valid_x[name], param.lower)
                    if hasattr(param, "upper"):
                        valid_x[name] = min(valid_x[name], param.upper)
                else:
                    valid_x[name] = param.default_value
        return valid_x

    def __call__(self, x) -> float:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
