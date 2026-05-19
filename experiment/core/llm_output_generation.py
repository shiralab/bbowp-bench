import json
import logging
import os

from openai import OpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_structured_output[T: BaseModel](
    model: str,
    system_prompt: str,
    user_prompt: str,
    output_model_class: type[T],
    seed: int,
) -> T:
    system_prompt = system_prompt.replace(
        "{SCHEMA}", json.dumps(output_model_class.model_json_schema(), indent=2)
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key is None or api_key.strip() == "":
        raise ValueError("OPENROUTER_API_KEY environment variable is not set or empty.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"},
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=seed,
        )

        if not response.choices:
            raise ValueError("No choices returned in the response.")
        choice = response.choices[0]
        output_text = choice.message.content
        parsed_output = output_model_class.model_validate_json(output_text)
    except ValidationError as e:
        logger.error(f"Validation error when parsing model output: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during structured output generation: {e}")
        raise

    return parsed_output
