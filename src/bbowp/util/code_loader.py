import importlib.util
import pathlib


def get_run_optimization(code_path: pathlib.Path):
    module_name = "run_optimization"

    spec = importlib.util.spec_from_file_location(module_name, code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.run_optimization
