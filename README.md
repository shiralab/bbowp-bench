# BBOWP-Bench

## Paper

This repository provides the code and benchmark suite for:

**BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems**  
Yutaro Yamada, Kei Hiroshima, Nozomu Yoshinari, Kento Uchida, and Shinichi Shirakawa  
*The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026), Evaluations and Datasets Track*  
[arXiv:2608.02612](https://arxiv.org/abs/2608.02612)

If you use BBOWP-Bench in your research, please cite our paper:

```bibtex
@inproceedings{yamada2026bbowp,
  title         = {{BBOWP-Bench}: Evaluating LLMs on Black-Box Optimization Word Problems},
  author        = {Yamada, Yutaro and Hiroshima, Kei and Yoshinari, Nozomu and Uchida, Kento and Shirakawa, Shinichi},
  booktitle     = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year          = {2026},
  eprint        = {2608.02612},
  archivePrefix = {arXiv}
}
```

## Tasks

| **Source** | **Task** | **Scenario** | **Dimension** | **Float** | **Int** | **Cat** |
|---|---|---|---|---|---|---|
| YAHPO | 1 | HPO of MLP | 7 | ✅ | ✅ | |
| | 2 | HPO of RcppHNSW | 6 | | ✅ | ✅ |
| | 3 | HPO of Ranger | 8 | ✅ | ✅ | ✅ |
| | 4 | HPO of RPART | 5 | ✅ | ✅ | ✅ |
| | 5 | HPO of SVM | 6 | ✅ | ✅ | ✅ |
| | 6 | HPO of XGBoost | 14 | ✅ | ✅ | ✅ |
| | 7 | Algorithm selection, HPO | 36 | ✅ | ✅ | ✅ |
| Olympus | 8 | N-benzylation | 4 | ✅ | | |
| | 9 | Color mixing BOB | 5 | ✅ | | |
| | 10 | Color mixing N9 | 3 | ✅ | | |
| | 11 | Fullerenes synthesis | 3 | ✅ | | |
| | 12 | Photo PCE10 blend | 4 | ✅ | | |
| | 13 | Photo WF3 blend | 4 | ✅ | | |
| | 14 | S $_N$ Ar reaction | 4 | ✅ | | |
| | 15 | HPLC system | 6 | ✅ | | |
| | 16 | Suzuki reaction | 4 | ✅ | | |
| MECHBench | 17 | Star-shaped crash box | 5 | ✅ | | |
| | 18 | Three-point bending of layered beam | 10 | ✅ | | |
| MuJoCo | 19 | Inverted Pendulum | 5 | ✅ | | |
| | 20 | Inverted Double Pendulum | 10 | ✅ | | |
| | 21 | Hopper | 36 | ✅ | | |
| | 22 | Reacher | 22 | ✅ | | |
| | 23 | Swimmer | 18 | ✅ | | |

- YAHPO Gym [Pfisterer, et al. 2022] ([GitHub](https://github.com/slds-lmu/yahpo_data/tree/v1.0.2?tab=readme-ov-file))
  - We extracted 7/14 tasks for the BBOWP dataset: lcbench, rbv2_aknn, rbv2_ranger, rbv2_rpart, rbv2_super, rbv2_svm, rbv2_xgboost.
    - For each task, we set the `task_id` to the first element of the instance list.
      - Instance corresponds to dataset (used to train the model and evaluate).
    - rbv2_glmnet is excluded due to the ambiguity of the variable names ("alpha" represents the L1/L2 ratio in this task, whereas it generally refers to the coefficient of the penalty terms).

- Olympus [Häse et al. 2021, Hickman et al. 2023] ([GitHub](https://github.com/aspuru-guzik-group/olympus))
  - We extracted 9/10 tasks for the BBOWP dataset: colors bob, colors n9, fullerenes, hplc, benzylation, photo pce10, photo wf3, snar, suzuki.
    - alkox is excluded due to the implementation error in the original.
  - The emulator model is fixed to BayesNeuralNet for all tasks.

- MECHBench [Rodríguez, I. O., et al. 2025] ([GitHub](https://github.com/BayesOptApp/MECHBench?tab=MIT-1-ov-file))
  - We extracted 2/3 tasks for the BBOWP dataset: Star-Shaped Crash Box and Three Point Bending of Layered Beam.
    - The dimension $d$ of the search space is fixed as follows: $d=5$ for Star-Shaped Crash Box and $d=10$ for Three-Point Bending of Layered Beam.
    - Long Crash Tube is excluded as we confirmed there is a discrepancy between paper and official code.
  - Python files and radioss files in `benchmark/eval_env/MECHBench/lib_mechbench` are copied from the official repository. (February 11, 2026)

- MuJoCo [Todorov, et al. 2012] ([GitHub](https://github.com/google-deepmind/mujoco))
  - We provide five black-box optimization tasks based on MuJoCo continuous control environments, accessed through the Gymnasium interface. In each task, the objective is to maximize the cumulative reward over a fixed-length episode by optimizing the parameters of a linear feedback policy. The design variables consist of feedback gains and bias terms of the linear policy.
    | Task | Environment ID |
    |---|---|
    | Inverted Pendulum | `InvertedPendulum-v5` |
    | Inverted Double Pendulum | `InvertedDoublePendulum-v5` |
    | Hopper | `Hopper-v5` |
    | Reacher | `Reacher-v5` |
    | Swimmer | `Swimmer-v5` |

## Usage
### Installation
1. Clone this repository

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already installed

3. Sync the environment (install packages)
```bash
uv sync
```

- To use SMAC3 as an optimization algorithm, install the optional `smac` extra instead:
```bash
uv sync --extra smac
```
  - `smac` depends on [pyrfr](https://github.com/automl/random_forest_run), which is built from source and requires [SWIG](https://www.swig.org/) and a C++ compiler.
  - With GCC 15 or later (e.g., Ubuntu 26.04), building pyrfr fails. In that case, run:
```bash
CXXFLAGS="-Wno-template-body" uv sync --extra smac
```

### Run LLM for search space design and algorithm selection
Our code uses [OpenRouter](https://openrouter.ai/) as the LLM API provider.

1. Place API key of [OpenRouter](https://openrouter.ai/) in `.env`

- Create `.env` file and add `OPENROUTER_API_KEY=`

```bash
echo 'OPENROUTER_API_KEY=YOURAPIKEY' > .env
```

2. Provide the task description to the LLM for search space design and algorithm selection

```bash
uv run experiment/run_bbowp_by_llm.py --exp_config experiment/config/bbowp_generation_config.yaml
```

- Edit [`experiment/config/bbowp_generation_config.yaml`](experiment/config/bbowp_generation_config.yaml) to set up experiments including LLM models, tasks, budget settings, difficulty settings, and seeds
  - By default, it runs for all 23 tasks, all 2 budget settings, all 3 difficulty settings, and one seed
- It generates `designed_answer.json` including the designed search space and the selected algorithm, under `result_dir` in [`experiment/config/bbowp_generation_config.yaml`](experiment/config/bbowp_generation_config.yaml)
- Optional: Set `--num_workers` for parallel execution
  - By default, it sets to `--num_workers=1`

### Run optimization
1. Run optimization on the designed search space and selected algorithm

```bash
uv run experiment/run_optimization.py --exp_config experiment/config/run_optimization_config.yaml
```

- Edit [`experiment/config/run_optimization_config.yaml`](experiment/config/run_optimization_config.yaml) to set up experiments including tasks, budget settings, difficulty settings, and seeds
  - By default, it runs for all 23 tasks, all 2 budget settings, all 3 difficulty settings, and one seed
- If there is no `designed_answer.json` at the expected path, it skips the setting
- It generates `optimization_result.json` including the best achieved solution and its score, under the same directory as `designed_answer.json`.
- Optional: Set `--num_workers` for parallel execution
  - By default, it sets to `--num_workers=1`
  - Note: Parallel execution can sometimes increase the overall execution time. We recommend starting with a small value for `--num_workers`.

### Run optimization using human baseline
1. To run optimization using human baseline, execute:

```bash
uv run experiment/run_optimization_baseline.py --exp_config experiment/config/run_optimization_baseline_config.yaml
```

- Edit [`experiment/config/run_optimization_baseline_config.yaml`](experiment/config/run_optimization_baseline_config.yaml) to set up experiments including tasks, budget settings, difficulty settings, and seeds
  - By default, it runs for all 23 tasks, all 2 budget settings, all 3 difficulty settings, and one seed
- First it generates `designed_answer.json` from `task.json` in our datasets `benchmark/config`
- Then it runs optimization and generates `optimization_result.json` including the best achieved solution and its score, under the same directory as `designed_answer.json`.
- Optional: Set `--num_workers` for parallel execution

### Evaluation: FRS
1. Aggregate the optimization results
```bash
uv run experiment/eval/fval/aggregate_score.py --config experiment/eval/fval/aggregate_config.yaml
```

2. Calculate the FRS score and save to csv
```bash
uv run experiment/eval/frs/calc_frs.py --config experiment/eval/frs/calc_frs_config.yaml
```

3. (Optional) Convert the csv to LaTeX table
```bash
uv run experiment/eval/frs/frs_csv2tex.py --csv_path (csv_path)
```

### Evaluation: EPDF
- **For quick evaluation**: run the following command (without any change) to plot EPDF in the paper:
```bash
uv run --with matplotlib experiment/eval/epdf/plot_search_space_sampling.py --exp_config experiment/eval/epdf/plot_search_space_sampling.yaml
```

1. To sample the points from the canonical domain, run the following command:
```bash
uv run experiment/eval/epdf/run_sample_point_domain.py --exp_config experiment/eval/epdf/sample_point_domain_config.yaml
uv run experiment/eval/epdf/extract_score_sample_point_domain.py --task_name (task_name) --result_dir experiment/eval/epdf/sample_point_domain
```

2. To sample the points from the designed search space, run the following command:
```bash
uv run experiment/eval/epdf/run_sample_point_searchspace.py --exp_config experiment/eval/epdf/sample_point_searchspace_config.yaml
uv run experiment/eval/epdf/extract_score_sample_point_searchspace.py --task_name (task_name) --result_dir experiment/result/example --out_dir experiment/eval/epdf/sample_point_searchspace
```

3. To calculate (plot) the EPDF, run the following command:
```bash
uv run --with matplotlib experiment/eval/epdf/plot_search_space_sampling.py --exp_config experiment/eval/epdf/plot_search_space_sampling.yaml
```

## License

Original code and documentation authored for BBOWP are licensed under the MIT License ([LICENSE](LICENSE)).

The BBOWP benchmark data under `benchmark/config` is licensed under the Creative Commons Attribution
4.0 International License (CC BY 4.0; [LICENSE-DATA](LICENSE-DATA)).

Third-party materials included in this repository remain subject to their own licenses and notices.

## Citation
- [Pfisterer, et al. 2022] Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M., & Bischl, B. (25--27 Jul 2022). YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. In I. Guyon, M. Lindauer, M. van der Schaar, F. Hutter, & R. Garnett (Eds), Proceedings of the First International Conference on Automated Machine Learning (p. 3/1-39). [Paper.](https://proceedings.mlr.press/v188/pfisterer22a.html)
- [Häse et al. 2021] Häse, Florian, Matteo Aldeghi, Riley J. Hickman, Loïc M. Roch, Melodie Christensen, Elena Liles, Jason E. Hein, and Alán Aspuru-Guzik. 2021. “Olympus: A Benchmarking Framework for Noisy Optimization and Experiment Planning.” Machine Learning: Science and Technology 2 (3): 035021. [Paper.](https://iopscience.iop.org/article/10.1088/2632-2153/abedc8/meta)
- [Hickman et al. 2023] Hickman, Riley, Priyansh Parakh, Austin Cheng, Qianxiang Ai, Joshua Schrier, Matteo Aldeghi, and Alán Aspuru-Guzik. 2023. “Olympus, Enhanced: Benchmarking Mixed-Parameter and Multi-Objective Optimization in Chemistry and Materials Science.” ChemRxiv. [Paper.](https://doi.org/10.26434/chemrxiv-2023-74w8d)
- [Rodríguez, I. O., et al. 2025] Rodríguez, I. O., Santoni, M. L., Duddeck, F., Doerr, C., Bäck, T., & Raponi, E. (2025). MECHBench: A Set of Black-Box Optimization Benchmarks originated from Structural Mechanics. arXiv [Cs.NE]. [Paper.](http://arxiv.org/abs/2511.10821)
- [Todorov, et al. 2012] Todorov, Emanuel, Tom Erez, and Yuval Tassa. 2012. “MuJoCo: A Physics Engine for Model-Based Control.” In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 5026–33. IEEE. [Paper.](https://ieeexplore.ieee.org/abstract/document/6386109)
