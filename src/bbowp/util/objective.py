import requests

from bbowp.schema.bbowp import BBOWPEvalConfig, Direction
from bbowp.schema.variable_type import VariableType
from bbowp.util.mapper import map_choice_names, map_variable_name


class Objective:
    def __init__(self, task: BBOWPEvalConfig, host: str, port: int, logger):
        self.task = task
        self.api_url = f"http://{host}:{port}/evaluate/{self.task.problem_id}"
        self.logger = logger
        self.worst_value = -float("inf")
        self.n_calls = 0
        if self.task.target.direction == Direction.MINIMIZE:
            self.worst_value = float("inf")

    def __call__(self, params: dict, preprocess_name_fn=None) -> float:
        self.n_calls += 1

        mapped_args = {}
        objective_args = self.task.execution.arguments

        for arg in objective_args:
            type_ = arg.type
            actual_name = arg.actual_name
            default_val = arg.default
            low_val = arg.low
            high_val = arg.high
            found_val = None

            if actual_name not in arg.aliases:
                aliases = (actual_name,) + arg.aliases
            else:
                aliases = tuple(arg.aliases)

            # resolve aliases of variable name
            mapped_names = map_variable_name(
                aliases, params, preprocess_name_fn=preprocess_name_fn
            )
            if len(mapped_names) == 0:
                self.logger.debug(
                    f"[Objective] No argument provided for variable '{actual_name}'. Using default value: '{actual_name}={default_val}'."
                )
                mapped_args[actual_name] = (
                    default_val  # no value provided for the argument, use default value
                )
            elif len(mapped_names) > 1:
                self.logger.debug(
                    f"[Objective] Ambiguous arguments. Multiple aliases {mapped_names} provided for variable '{actual_name}'. Returning worst value."
                )
                return self.worst_value
            elif len(mapped_names) == 1:
                (mapped_name_for_arg,) = mapped_names
                found_val = params[mapped_name_for_arg]

                # cast found_val to the correct type
                try:
                    if type_ == VariableType.INT:
                        found_val = int(found_val)
                    elif type_ == VariableType.FLOAT:
                        found_val = float(found_val)
                    elif type_ == VariableType.BOOL:
                        found_val = bool(found_val)
                    elif type_ == VariableType.CATEGORICAL:
                        found_val = str(found_val)
                except Exception:
                    self.logger.debug(
                        f"[Objective] Could not cast value '{found_val}' of variable '{actual_name}' to type '{type_}'. Returning worst value."
                    )
                    return self.worst_value

                # check whether found_val is within the domain
                if (low_val is not None and found_val < low_val) or (
                    high_val is not None and found_val > high_val
                ):
                    self.logger.debug(
                        f"[Objective] Value '{found_val}' of variable '{actual_name}' is outside the domain [{low_val}, {high_val}]. Returning worst value."
                    )
                    return self.worst_value

                # resolve aliases of categorical choices
                if type_ == VariableType.CATEGORICAL:
                    choice_mapped_names = map_choice_names(
                        found_val, arg.choices, preprocess_name_fn=preprocess_name_fn
                    )
                    if len(choice_mapped_names) == 0:
                        self.logger.debug(
                            f"[Objective] Invalid categorical choice '{found_val}' for variable '{actual_name}'. Returning worst value."
                        )
                        return self.worst_value
                    elif len(choice_mapped_names) > 1:
                        self.logger.debug(
                            f"[Objective] Ambiguous categorical choice '{found_val}' for variable '{actual_name}'. Multiple mapped choices: {choice_mapped_names}. Returning worst value."
                        )
                        return self.worst_value
                    elif len(choice_mapped_names) == 1:
                        (mapped_choice_name,) = choice_mapped_names
                        mapped_args[actual_name] = mapped_choice_name
                else:
                    mapped_args[actual_name] = found_val

        response = requests.post(self.api_url, json={"params": mapped_args})

        if response.status_code != 200:
            self.logger.error(f"[Objective] Server Error: {response.text}")
            self.logger.error(f"[Objective] status_code: {response.status_code}")
            raise RuntimeError(
                f"Objective execution failed with status code {response.status_code}"
            )

        response_json = response.json()

        self.logger.info(
            f"[Objective] number of evaluations: {self.n_calls}, sampled parameters: {mapped_args}, value of objective: {response_json['score']}"
        )

        return float(response_json["score"])
