import json
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

app = FastAPI()

TASK_CONFIGS = {
    "Hopper": "Hopper-v5",
    "Inverted_Pendulum": "InvertedPendulum-v5",
    "Inverted_Double_Pendulum": "InvertedDoublePendulum-v5",
    "Reacher": "Reacher-v5",
    "Swimmer": "Swimmer-v5",
}


class EvalRequest(BaseModel):
    params: dict[str, Any]


def evaluate_policy(
    W: np.ndarray,
    b: np.ndarray,
    task: str,
    n_eval_episodes: int,
    seed: int = 0,
) -> float:
    env = gym.make(task)
    total_reward = 0.0
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = np.clip(W @ obs + b, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    env.close()
    return total_reward / n_eval_episodes


def load_params(task_name: str) -> (dict[str, Any], dict[str, Any]):
    with open(f"/app/params/{task_name}.json", "r") as f:
        params = json.load(f)
    return params["obs"], params["act"]


def get_policy_params(
    params: dict[str, Any],
    task_name: str,
    W_init: np.ndarray,
    b_init: np.ndarray,
):
    """
    load param list and map to W and b, for vacant params, use values from W_init and b_init
    """
    obs_params, act_params = load_params(task_name)
    if (
        len(obs_params) != W_init.shape[1]
        or len(act_params) != W_init.shape[0]
        or len(act_params) != b_init.shape[0]
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Dimension mismatch between provided parameters and expected policy dimensions.({len(obs_params)} obs params, {len(act_params)} act params, W_init shape: {W_init.shape}, b_init shape: {b_init.shape})",
        )

    W = pd.DataFrame(W_init, index=act_params, columns=obs_params)
    b = pd.Series(b_init, index=act_params)

    for param_name, value in params.items():
        obs, act = param_name.split("_to_")
        if obs in obs_params and act in act_params:
            W.loc[act, obs] = value
        elif obs == "bias" and param_name in act_params:
            b.loc[param_name] = value
        else:
            logger.warning(
                f"Parameter '{param_name}' does not match any expected format."
            )
    return np.array(W), np.array(b)


@app.post("/evaluate/{task_name}")
def evaluate(task_name: str, req: EvalRequest):
    n_eval_episodes = 10
    seed_simulation = 42  # should be unfixed perhaps?

    if "_" not in task_name:
        raise HTTPException(
            status_code=400, detail=f"Invalid task name format: '{task_name}'"
        )
    task_name_ = task_name[
        task_name.index("_") + 1 :
    ]  # remove prefix, e.g. "MuJoCo_Hopper" -> "Hopper"

    try:
        env = gym.make(TASK_CONFIGS[task_name_])
        n_dim_obs = env.observation_space.shape[0]
        n_dim_act = env.action_space.shape[0]
        env.close()

        W_init = np.zeros((n_dim_act, n_dim_obs))  # could be random or something else
        b_init = np.zeros(n_dim_act)  # could be random or something else
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unsupported task: '{task_name_}'")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"param file for Task '{task_name_}' was not found."
        )

    try:
        W, b = get_policy_params(req.params, task_name_, W_init, b_init)

        result = evaluate_policy(
            W=W,
            b=b,
            task=TASK_CONFIGS[task_name_],
            n_eval_episodes=n_eval_episodes,
            seed=seed_simulation,
        )
        return {"score": float(result)}

    except HTTPException:
        raise  # re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
