import logging
import pathlib

from dotenv import load_dotenv
from pydantic import ValidationError

from bbowp.schema.bbowp import BBOWP
from bbowp.schema.designed_answer import DesignedAnswer

from .llm_output_generation import generate_structured_output

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_task_config(task_config_path: str | pathlib.Path) -> BBOWP:
    task_config_path = pathlib.Path(task_config_path)
    try:
        task_json_str = task_config_path.read_text(encoding="utf-8")
    except IsADirectoryError:
        logger.error(
            f"Task config path is a directory, expected a file: {task_config_path}"
        )
        raise
    except FileNotFoundError:
        logger.error(f"Task config file not found: {task_config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading task config: {e}")
        raise

    try:
        task_config = BBOWP.model_validate_json(task_json_str)
    except ValidationError as e:
        logger.error(f"Validation error in task config: {e}")
        raise

    return task_config


def generate_designed_answer(
    task_desc: str,
    model: str,
    seed: int,
    prompt: dict,
    example: dict[str, str] | None = None,
) -> DesignedAnswer:
    system_prompt = prompt.get("system_prompt", "")
    user_prompt = prompt.get("user_prompt", "")
    user_prompt = user_prompt.format(TASK_DESCRIPTION=task_desc)

    # incorporate example into system prompt if provided (few-shot)
    if example is not None:
        system_prompt = system_prompt.replace(
            "{EXAMPLE_TASK_DESCRIPTION}", example["EXAMPLE_TASK_DESCRIPTION"]
        ).replace("{EXAMPLE_TASK_OUTPUT}", example["EXAMPLE_TASK_OUTPUT"])

    designed_answer = generate_structured_output(
        model, system_prompt, user_prompt, DesignedAnswer, seed
    )

    return designed_answer


def execute_designed_answer_generation(
    task_config_path: str | pathlib.Path,
    diff: str,
    budget: str,
    prompt: dict,
    model: str,
    out_dir: str | pathlib.Path,
    seed: int = 42,
    example_task_config_path: str | pathlib.Path | None = None,
):
    # set up output path
    output_path = pathlib.Path(out_dir) / "designed_answer.json"
    if output_path.exists():  # avoid overwriting existing output
        logger.debug(f"Output file already exists. Skipping generation: {output_path}")
        return

    # load path of task.json
    task_config = get_task_config(task_config_path)

    # extract task description for the specified difficulty
    task_desc = next((d for d in task_config.description if d.difficulty == diff), None)
    if task_desc is None:
        available_diffs = ", ".join(d.difficulty for d in task_config.description)
        raise ValueError(
            f"No description found for difficulty '{diff}'. Available difficulties: {available_diffs}"
        )
    task_desc = task_desc.text

    # extract max_eval_value for the specified budget
    budget_amount = next((b for b in task_config.budget if b.amount == budget), None)
    if budget_amount is None:
        available_budgets = ", ".join(b.amount for b in task_config.budget)
        raise ValueError(
            f"Unknown budget '{budget}'. Available budgets: {available_budgets}"
        )
    max_eval_value = budget_amount.max_eval_value

    task_desc = task_desc.format(MAX_EVAL_VALUE=max_eval_value)

    # load example task config if provided
    example = None
    if example_task_config_path is not None:
        example_task_config = get_task_config(example_task_config_path)
        example_task_desc = next(
            (d for d in example_task_config.description if d.difficulty == diff), None
        )
        example_task_answer = next(
            (a.answer_item for a in example_task_config.answer if a.amount == budget),
            None,
        )
        example_task_desc = example_task_desc.text
        example_task_desc = example_task_desc.format(MAX_EVAL_VALUE=max_eval_value)
        example_task_answer = example_task_answer.model_dump()
        example_designed_answer = DesignedAnswer(**example_task_answer)
        example = {
            "EXAMPLE_TASK_DESCRIPTION": example_task_desc,
            "EXAMPLE_TASK_OUTPUT": example_designed_answer.model_dump_json(indent=4),
        }

    # generate designed answer
    load_dotenv()
    designed_answer = generate_designed_answer(task_desc, model, seed, prompt, example)

    # save designed answer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(designed_answer.model_dump_json(indent=2))
