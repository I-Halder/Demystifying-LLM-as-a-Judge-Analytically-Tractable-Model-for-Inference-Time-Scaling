# Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling

> **Authors:** Indranil Halder, Cengiz Pehlevan  
> **Institution:** Harvard John A. Paulson School of Engineering and Applied Sciences  
> **Paper:** [arXiv:2512.19905](https://arxiv.org/abs/2512.19905)

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [LLM Experiments](#llm-experiments)
    - [compute_delta_vs_k.py](#compute_delta_vs_kpy)
    - [compute_delta_vs_T.py](#compute_delta_vs_tpy)
    - [SLURM Example](#slurm-example)
- [Code Structure](#code-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Issues](#issues)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for judge-model inference
- Sufficient GPU memory for the selected base model and judge model

### Setup

1. Clone the repository:

```bash
git clone https://github.com/I-Halder/Demystifying-LLM-as-a-Judge-Analytically-Tractable-Model-for-Inference-Time-Scaling.git
cd Demystifying-LLM-as-a-Judge-Analytically-Tractable-Model-for-Inference-Time-Scaling
```

2. Create and activate an environment:

```bash
conda create -n inference-time-scaling python=3.10
conda activate inference-time-scaling
```

3. Install dependencies:

```bash
pip install -e .[hf]
pip install vllm numpy matplotlib requests
```

---

## Usage

### LLM Experiments

The repository includes two experiment drivers for measuring how the aggregate error term `δ` changes under inference-time scaling:

- `compute_delta_vs_k.py` sweeps the number of candidate completions `k` while holding selection temperature `T` fixed.
- `compute_delta_vs_T.py` sweeps reward temperature `T` while holding the number of samples per prompt `k` fixed.


#### Judge backends

Both experiment drivers use the same two judge backends:

- **Generative judge** via `AutoModelForCausalLM`, which generates a scalar score in the range `[0, 10]`
- **PRM-style judge** via `AutoModel`, which scores `<extra_0>` step separators and converts the mean positive-step probability to a score in `[0, 10]`

Judge mode is auto-detected from the model name. If the judge model identifier contains `prm`, the PRM path is used; otherwise the generative path is used. The `compute_delta_vs_T.py` driver also exposes `--judge_mode` to override this behavior explicitly.

**Important**

 - Both scripts modify the task YAML file in place by rewriting `repeats: <k>` before sample generation.

---

#### compute_delta_vs_k.py


##### Command-line arguments

| Argument | Required | Description |
|---|---:|---|
| `--T` | Yes | Selection temperature used to compute `δ` |
| `--k_min` | No | Minimum `k` value when constructing a range |
| `--k_max` | No | Maximum `k` value when constructing a range |
| `--k_step` | No | Step size for the `k` sweep |
| `--k_values` | No | Comma-separated explicit `k` values; overrides range construction |
| `--model_args` | No | vLLM model arguments forwarded to `lm_eval` |
| `--tasks` | No | `lm_eval` task name |
| `--output_path` | No | Base output directory prefix; generated runs are written to `<output_path>_k<k>` |
| `--limit` | No | Optional dataset fraction passed to `lm_eval` |
| `--yaml_path` | No | Path to the task YAML file containing `repeats` |
| `--skip_lm_eval` | No | Reuse existing sample files instead of generating new ones |
| `--samples_pattern` | No | Glob pattern for sample files; may include `{k}` as a placeholder |
| `--judge_model` | No | Hugging Face model ID for the local judge |
| `--plot_output` | No | Output filename for the final `δ` vs `k` plot |
| `--epsilon` | No | Gaussian noise scale applied to judge scores |
| `--seed` | No | Random seed for score perturbation |

##### Range behavior

- If `--k_values` is provided, it fully overrides `--k_min`, `--k_max`, and `--k_step`.
- Otherwise, the script evaluates `list(range(k_min, k_max + 1, k_step))`, so `k_max` is effectively **inclusive** when it falls on the step schedule.
- When `--skip_lm_eval` is omitted, the script launches a separate `lm_eval` run for each requested `k`.
- `T = 0` is supported and corresponds to hard best-of-`k` selection by judge score.

##### Example: generate fresh samples for each `k`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python compute_delta_vs_k.py \
  --T 10.0 \
  --k_min 1 \
  --k_max 128 \
  --k_step 4 \
  --model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9' \
  --tasks gsm8k_cot_self_consistency \
  --yaml_path lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml \
  --output_path out/gsm8k_k_sweep \
  --judge_model mistralai/Mistral-7B-Instruct-v0.3 \
```

##### Example: reuse existing samples

```bash
python compute_delta_vs_k.py \
  --T 10.0 \
  --k_values 4,8,16,32,64 \
  --skip_lm_eval \
  --samples_pattern 'out/gsm8k_k_sweep_k{k}/**/samples_*.jsonl' \
  --judge_model mistralai/Mistral-7B-Instruct-v0.3
```

---

#### compute_delta_vs_T.py


##### Command-line arguments

| Argument | Required | Description |
|---|---:|---|
| `--samples_jsonl` | Conditionally | Path or glob pattern for existing sample files |
| `--generate_samples` | Conditionally | Generate samples with `lm_eval` before computing `δ(T)` |
| `--k` | No | Number of samples per prompt when generating data |
| `--model_args` | No | vLLM model arguments forwarded to `lm_eval` |
| `--tasks` | No | `lm_eval` task name |
| `--output_path` | No | Output directory used by `lm_eval` generation |
| `--yaml_path` | No | Path to the task YAML file containing `repeats` |
| `--limit` | No | Optional dataset fraction passed to `lm_eval` |
| `--T_values` | No | Comma-separated explicit temperature values; overrides range construction |
| `--T_min` | No | Lower bound of the temperature sweep |
| `--T_max` | No | Upper bound of the temperature sweep |
| `--T_step` | No | Step size used to build temperatures from `T_min` to `T_max` (float) |
| `--reward` | No | Reward provider: `judge` or `http` |
| `--reward_api` | No | URL for the HTTP reward endpoint when `--reward http` is used |
| `--judge_model` | No | Hugging Face model ID for the local judge |
| `--judge_mode` | No | Judge mode: `auto`, `gen`, or `prm` |
| `--epsilon` | No | Gaussian noise scale applied to judge scores |
| `--seed` | No | Random seed for score perturbation |

##### Range behavior

- If `--T_values` is provided, the script uses those values directly.
- Otherwise, temperatures are constructed with `np.arange(T_min, T_max + 0.5*T_step, T_step)` and then converted to a list. This makes the upper bound effectively inclusive when it falls on the step schedule.
- `T = 0` is supported and corresponds to hard best-of-`k` selection by judge score.

##### Example: generate samples and sweep temperature locally

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python compute_delta_vs_T.py \
  --generate_samples \
  --k 64 \
  --T_min 0.0 \
  --T_max 10.0 \
  --T_step 1.0 \
  --model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9' \
  --tasks gsm8k_cot_self_consistency \
  --yaml_path lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml \
  --output_path out/gsm8k_T_sweep \
  --judge_model mistralai/Mistral-7B-Instruct-v0.3
```

##### Example: reuse existing samples

```bash
python compute_delta_vs_T.py \
  --samples_jsonl 'out/gsm8k_T_sweep/**/samples_*.jsonl' \
  --k 64 \
  --T_values 0,0.5,1,2,5,10 \
  --judge_model mistralai/Mistral-7B-Instruct-v0.3
```


### SLURM Example

The following examples match the current CLI exposed by the uploaded scripts.

```bash
#!/bin/bash
#SBATCH --job-name=delta_sweeps
#SBATCH --output=delta_sweeps-%j.out
#SBATCH --cpus-per-task=18
#SBATCH --mem=72G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4

module load python/3.10.13-fasrc01
module load Anaconda2/2019.10-fasrc01
conda activate inference-time-scaling

export HF_TOKEN="..."
export HF_HOME=/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python compute_delta_vs_k.py \
  --T 10.0 \
  --k_values 4,8,16,32,64 \
  --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9" \
  --tasks gsm8k_cot_self_consistency \
  --yaml_path lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml \
  --output_path out/meta-llama_T10 \
  --judge_model mistralai/Mistral-7B-Instruct-v0.3 \
  --plot_output delta_vs_k.png

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python compute_delta_vs_T.py \
  --generate_samples \
  --k 64 \
  --T_min 0.0 \
  --T_max 10.0 \
  --T_step 1.0 \
  --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9" \
  --tasks gsm8k_cot_self_consistency \
  --yaml_path lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml \
  --output_path out/meta-llama_k64 \
  --judge_model mistralai/Mistral-7B-Instruct-v0.3
```

---

## Code Structure

```text
.
├── BLR_zero_T.py
├── BLR_non_zero_T.py
├── compute_delta_vs_k.py
├── compute_delta_vs_T.py
├── pyproject.toml
├── README.md
├── README_updated.md
├── lm_eval/
│   ├── tasks/gsm8k/
│   │   └── gsm8k-cot-self-consistency.yaml
│   ├── models/vllm_causallms.py
│   └── evaluator.py
└── LICENSE
```

---


## Citation

If you use this code or find this work useful, please cite:

```bibtex
@misc{halder2025demystifyingllmasajudgeanalyticallytractable,
  title={Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling},
  author={Indranil Halder and Cengiz Pehlevan},
  year={2025},
  eprint={2512.19905},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.19905}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Funding: DARPA grant AIQ-HR00112520041, NSF CAREER Award IIS-2239780, Simons Collaboration on the Physics of Learning, and the Kempner Institute for the Study of Natural and Artificial Intelligence
- Code base: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) by EleutherAI

---

## Issues

- For bug reports, open a GitHub issue
- For questions about the theoretical analysis, contact `ihalder@g.harvard.edu`
