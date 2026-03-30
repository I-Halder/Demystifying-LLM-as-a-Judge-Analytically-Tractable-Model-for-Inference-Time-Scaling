import os # For operating system dependent functionality like file paths
import argparse # For parsing command-line arguments
import sys
import glob # For file pattern matching
import json # For working with JSON data
import re # regular expressions for parsing strings
import subprocess # For running external commands
from typing import List, Tuple, Dict, Iterable, Optional # For type hinting
from functools import lru_cache # For caching function input, output pairs so that when an input is encoutered but appeared before, it skips the computation and returns the answer from the lookup dict
import requests # For making HTTP requests

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, logging, AutoModel
logging.set_verbosity_error()
        
n_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpus}")

# Enable TF32 for H100 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Parsing the answer
RE_THE_ANSWER_IS=re.compile(r"The answer is (\-?[0-9\.\,]*[0-9]+)")
RE_HASHES=re.compile(r"####\s*(-?\d[\d,\.]*)")
RE_GENERIC_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")

def _patch_transformers_cache_api_for_prm() -> None:
    try:
        from transformers.cache_utils import DynamicCache  
    except Exception:
        return

    # Nothing to do on older transformers where the method exists.
    if hasattr(DynamicCache, "get_usable_length"):
        return
    if not hasattr(DynamicCache, "get_seq_length"):
        return

    def _get_usable_length(self, *args, **kwargs):
        layer_idx = 0
        if len(args) >= 2:
            layer_idx = args[1]
        elif "layer_idx" in kwargs:
            layer_idx = kwargs["layer_idx"]
        try:
            return self.get_seq_length(layer_idx)
        except TypeError:
            return self.get_seq_length()

    DynamicCache.get_usable_length = _get_usable_length  

@lru_cache(maxsize=131072)
def extract_num(text: str) -> Optional[str]:
    """
    Extracts a number from the given text using predefined regular expressions.
    """
    if not text:
        return None
    m = RE_THE_ANSWER_IS.search(text)
    if m:
        return m.group(1).replace(",","")
    m = RE_HASHES.search(text)
    if m:
        return m.group(1).replace(",", "")
    m = RE_GENERIC_NUM.search(text)
    if m:
        return m.group(1).replace(",","")
    return None

@lru_cache(maxsize=131072)
def exact_numeric_equal(pred_text: str, gold_text: str) -> int:
    """
    Compares the predicted text and gold text for exact numeric equality for the extracted text.   
    """
    pp, gg = extract_num(pred_text), extract_num(gold_text)
    if pp is None or gg is None:
        return 0
    try:
        return 1 if float(pp)==float(gg) else 0
    except:
        return 1 if pp.strip() == gg.strip() else 0

def set_repeats_in_yaml(yaml_path: str, k: int) -> None:
    """
    Modify the 'repeats' parameter in a YAML task configuration file.
    """
    cmd = f"sed -i 's/repeats: .*/repeats: {k}/' {yaml_path}"
    print(f"[set_repeats_in_yaml] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sed command failed: {result.stderr}")
    print(f"[set_repeats_in_yaml] Set repeats={k} in {yaml_path}")


def run_lm_eval(
    model_args: str,
    tasks: str,
    output_path: str,
    limit: Optional[float] = None
) -> str:
    """
    Execute the lm_eval harness with specified configuration.
    """
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--output_path", output_path,
        "--log_samples",
    ]
    # cmd = [
    #     "lm_eval",
    #     "--model", "vllm",
    #     "--model_args", model_args,
    #     "--tasks", tasks,
    #     "--output_path", output_path,
    #     "--log_samples",
    # ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    print(f"[run_lm_eval] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed with return code {result.returncode}")
    
    # Return glob pattern to find generated samples
    samples_glob = f"{output_path}/**/samples_*.jsonl"
    print(f"[run_lm_eval] Samples will be at: {samples_glob}")
    return samples_glob

def generate_samples(
    k: int,
    model_args: str,
    tasks: str,
    output_path: str,
    yaml_path: str,
    limit: Optional[float] = None
) -> str:
    """
    Generate samples by running lm_eval.
    """
    print(f"\n{'='*60}")
    print(f"[generate_samples] Generating samples with k={k}")
    print(f"{'='*60}")
    
    # Step 1: Set repeats value in YAML
    set_repeats_in_yaml(yaml_path, k)
    
    # Step 2: Run lm_eval
    samples_glob = run_lm_eval(model_args, tasks, output_path, limit)
    
    print(f"[generate_samples] Generation complete!")
    return samples_glob

class RewardProvider:
    """Abstract base class for reward scoring providers."""
    
    def score_batch(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        Score a batch of (question, COT, answer) triples.
        """
        raise NotImplementedError

class HttpReward(RewardProvider):
    """Reward scorer that queries an HTTP API endpoint."""
    
    def __init__(self, url: str, timeout: float = 30.0) -> None:
        """
        Initialize HTTP reward provider.
        """
        self.session = requests.Session()
        self.url, self.timeout = url, timeout
    
    def score_batch(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        Score triples by sending HTTP POST requests to the API.
        
        Args:
            triples (List[Tuple[str, str, str]]): List of (question, chain of thought, answer) tuples.
        
        Returns:
            torch.Tensor: 1D tensor of scores from the API.
        """
        out = []
        for q, cot, ans in triples:
            r = self.session.post(self.url, json={"prompt": q, "cot": cot, "final_answer": ans}, timeout=self.timeout)
            r.raise_for_status()
            out.append(float(r.json()["score"]))
        return torch.tensor(out, dtype=torch.float32)

class JudgeReward(RewardProvider):
    
    def __init__(self, model_name: str, dtype: str = "bfloat16", mode: str = "auto", max_length: int = 1024, epsilon: float = 0.0, seed: Optional[int] = None) -> None:
        
        mode = (mode or "auto").lower()
        if mode not in {"auto", "gen", "prm"}:
            raise ValueError("--judge_mode must be one of: auto, gen, prm")

        if mode == "auto":
            self.mode = "prm" if re.search(r"\bprm\b", model_name, flags=re.IGNORECASE) else "gen"
        else:
            self.mode = mode

        # Load tokenizer with left padding
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        
        effective_device_map = "balanced" if n_gpus > 1 else "auto"
        
        self._model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": effective_device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if self.mode == "prm":
            _patch_transformers_cache_api_for_prm()
            self.model = AutoModel.from_pretrained(model_name, **self._model_kwargs).eval()
            # Disable KV caching
            try:
                self.model.config.use_cache = False
            except Exception:
                pass
            self._device = next(self.model.parameters()).device
            # Per the PRM model card, we insert <extra_0> after each reasoning step.
            # Rewards are computed from the probability that <extra_0> is classified as "positive".
            self.step_sep = "<extra_0>"
            step_ids = self.tok.encode(self.step_sep, add_special_tokens=False)
            # PRM expects <extra_0> to be a single token so we can mask its positions.
            if len(step_ids) != 1:
                raise RuntimeError(f"Expected '{self.step_sep}' to tokenize to 1 id, got {step_ids}")
            self.step_sep_id = step_ids[0]
            self.system_prompt = r"Please reason step by step, and put your final answer within \boxed{}."
        else:
            model_kwargs = dict(self._model_kwargs)
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
                print("[JudgeReward] Using Flash Attention 2")
            except Exception as e:
                print(f"[JudgeReward] Flash Attention 2 failed: {e}")
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
                print("[JudgeReward] Using standard attention")
        
            if hasattr(self.model, "hf_device_map"):
                devices_used = set(self.model.hf_device_map.values())
                print(f"[JudgeReward] Model sharded across: {devices_used}")
            
            self._num = re.compile(r"(-?\d+(?:\.\d+)?)")
            #self.prompt = "Score 0-10:\n[Q]{q}\n[A]{sol}\n[Final]{ans}\nScore:"
            self.prompt = "Score the answer of the student based on its logical consistency of reasoning and accuracy of the final answer. The output should be a number between 0 to 10.\n[question]{q}\n[reasoning of the student]{sol}\n[final answer of the student]{ans}\n[Score]:"
        
        self.max_length = max_length
        self._device = next(self.model.parameters()).device
        self.epsilon = epsilon # set to default value of 0.0 for backward compatibility; used to tune judge quality
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def _fmt_gen(self, q: str, sol: str, ans: str) -> str:
        """
        Format a (question, COT, answer) triple into a scoring prompt.
        """
        return self.prompt.format(
            q=(q or "")[:400].strip(),
            sol=(sol or "")[:600].strip(),
            ans=(ans or "")[:80].strip()
        )
        
    def _fmt_prm(self, q: str, sol: str, ans: str) -> str:
        """Build a chat-formatted string expected by PRM.
        """
        sol = (sol or "").strip()
        ans = (ans or "").strip()

        # Pattern: split after a period that ends a sentence containing '='
        sentences = re.findall(r'[^.]+(?:\.|$)', sol)

        steps = []
        current = ""
        for s in sentences:
            # preserve original spacing between sentences
            current += s.strip() + " "
            # if this sentence contains '=', treat it as a step boundary
            if "=" in s:
                steps.append(current.strip())
                current = ""
        # leftover text becomes last step
        if current.strip():
            steps.append(current.strip())

        if not steps:
            steps = [""]

        assistant_content = self.step_sep.join(steps) + self.step_sep  # ensure a trailing <extra_0> for the last step
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (q or "").strip()[:2000]},
            {"role": "assistant", "content": assistant_content[:8000]},
        ]
        # NOTE: tokenize=False => when tokenizing later, set add_special_tokens=False.
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    
    def _parse_batch(self, texts: List[str]) -> List[float]:
        """
        Parse numeric scores from judge model outputs.
        """
        scores = []
        for s in texts:
            print(f"[JudgeReward] Actual reward from the Judge: {s}")
            m = self._num.search(s or "")
            v = float(m.group(1)) if m else 0.0
            v = v + 10 * self.epsilon * self.rng.standard_normal()
            print(f"[JudgeReward] Randomized score: {v}")
            score_clipped = max(0.0, min(10.0, v))
            print(f"[JudgeReward] Clipped score: {score_clipped}")
            scores.append(score_clipped)
        return scores
    
    def score_batch(self, triples: List[Tuple[str, str, str]], batch_size: int = 256) -> torch.Tensor:
        """Scores a batch of (question, solution, answer) triples.
        """
        if not triples:
            return torch.empty((0,), dtype=torch.float32, device=self._device)

        if self.mode == "prm":
            return self._score_batch_prm(triples, batch_size=batch_size)
        return self._score_batch_gen(triples, batch_size=batch_size)
    
    def _score_batch_gen(self, triples: List[Tuple[str,str,str]], batch_size: int = 512, max_new_tokens: int = 4) -> torch.Tensor:
        """
        Score a batch of (question, COT, answer) triples.
        """
        all_scores = []
        n_batches = (len(triples) + batch_size - 1) // batch_size
        
        print(f"[JudgeReward] Scoring {len(triples)} triples in {n_batches} batches of {batch_size}.")
        
        for batch_idx in range(0, len(triples), batch_size):
            batch_triples = triples[batch_idx:batch_idx+batch_size]
            batch_texts = [self._fmt_gen(*t) for t in batch_triples]
            
            with torch.inference_mode():
                enc = self.tok(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length
                )
                enc = {k: v.to(self._device, non_blocking=True) for k, v in enc.items()}
                
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tok.pad_token_id,
                    use_cache=True,
                )
                
                input_lens = enc["input_ids"].shape[1] #enc["attention_mask"].sum(dim=1)
                decoded = []
                for j in range(len(batch_texts)):
                    # plen = input_lens[j].item()
                    plen = input_lens
                    decoded.append(self.tok.decode(gen[j][plen:], skip_special_tokens=True))
                
                all_scores.extend(self._parse_batch(decoded))
            
            if (batch_idx // batch_size + 1) % 10 == 0:
                print(f"  Batch {batch_idx // batch_size + 1}/{n_batches} done")
        
        return torch.tensor(all_scores, dtype=torch.float32, device=self._device)
    
    def _score_batch_prm(self, triples: List[Tuple[str, str, str]], batch_size: int) -> torch.Tensor:
        """PRM-scoring.
        """
        all_scores: List[float] = []
        n_batches = (len(triples) + batch_size - 1) // batch_size
        print(f"[score_batch:prm] Scoring {len(triples)} samples in {n_batches} batches of size {batch_size}")

        for batch_idx in range(0, len(triples), batch_size):
            batch_triples = triples[batch_idx : batch_idx + batch_size]
            conv_strs = [self._fmt_prm(q, sol, ans) for q, sol, ans in batch_triples]

            with torch.inference_mode():
                enc = self.tok(
                    conv_strs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )
                enc = {k: v.to(self._device, non_blocking=True) for k, v in enc.items()}

                try:
                    outputs = self.model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                        use_cache=False,
                    )
                except TypeError:
                    outputs = self.model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                # For PRM, logits should be [batch, seq_len, 2]
                if logits.dim() != 3 or logits.size(-1) != 2:
                    raise RuntimeError(
                        f"Unexpected PRM logits shape {tuple(logits.shape)}. "
                        "Make sure you loaded a correct PRM model."
                    )
                probs = F.softmax(logits.float(), dim=-1)
                pos_probs = probs[..., 1]

                attn = enc.get("attention_mask")
                if attn is None:
                    attn = torch.ones_like(enc["input_ids"], dtype=torch.long)
                step_mask = (enc["input_ids"] == self.step_sep_id) & attn.bool()
                # step_mask[b, t] = True exactly at the <extra_0> tokens that delimit steps.
                counts = step_mask.sum(dim=1).clamp_min(1)
                # Number of step separators per sample.
                mean_step_reward = (pos_probs * step_mask).sum(dim=1) / counts
                # Scalar PRM score per sample: mean probability that each <extra_0> is "positive".
                scores = (10.0 * mean_step_reward).clamp(0.0, 10.0) 
                all_scores.extend(scores.detach().cpu().tolist())

            if (batch_idx // batch_size + 1) % 20 == 0:
                print(f"  Batch {batch_idx // batch_size + 1}/{n_batches}")

        return torch.tensor(all_scores, dtype=torch.float32, device=self._device)

def iter_jsonl(glob_pat: str) -> Iterable[Dict]:
    """
    Iterates over JSON files matching the given glob pattern and returns JSON objects one at a time.
    """
    for path in glob.glob(glob_pat): # glob.glob returns a list of file paths matching the given pattern glob_pat
        with open(path, 'r') as fh:
            for line in fh:
                line = line.strip() # Remove leading/trailing whitespace and \n
                if line:
                    yield json.loads(line) # Instead of loading all JSON objects into memory at once, yield returns them one at a time

def plot_vary_T(
    T_values: List[float],
    k: float, 
    samples_glob: str, 
    reward_mode: str, 
    reward_api: Optional[str], 
    judge_model: Optional[str], 
    judge_mode: str,
    epsilon: float = 0.0, # updated 03/05 by AB
    seed: Optional[int] = None
) -> None:
    """
    Compute and plot delta as a function of temperature T.    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if reward_mode == "http":
        if not reward_api:
            raise ValueError("--reward http requires --reward_api URL")
        R = HttpReward(reward_api)
    else:
        R = JudgeReward(judge_model or "mistralai/Mistral-7B-Instruct-v0.3", mode=judge_mode, epsilon=epsilon, seed=seed)
    
    #R = JudgeReward(judge_model or "mistralai/Mistral-7B-Instruct-v0.3", epsilon=epsilon, seed=seed)
    
    # ========== Phase 1: Load all data ==========
    print("[plot_vary_T] Loading JSONL data.")
    all_triples = []
    all_r_Ts = []
    prompt_sizes = []
    n_empty = 0

    for rec in iter_jsonl(samples_glob):
        doc = rec.get("doc") or {}
        prompt = doc.get("question") or rec.get("prompt") or rec.get("inputs") or ""
        gold = doc.get("answer") or rec.get("target") or ""

        gens: List[str] = []
        if "resps" in rec:
            rs = rec["resps"]
            if isinstance(rs, list):
                if rs and isinstance(rs[0], list):
                    gens.extend([s for s in rs[0] if isinstance(s, str)])
                else:
                    gens.extend([s for s in rs if isinstance(s, str)])
        elif "outputs" in rec:
            for o in rec["outputs"]:
                text = o.get("text") or o.get("decoded") or o.get("output") or ""
                if text:
                    gens.append(text)

        if not gens:
            n_empty += 1
            continue

        prompt_sizes.append(len(gens))
        for g in gens:
            ans = extract_num(g) or ""
            r_T = exact_numeric_equal(g, gold)
            all_triples.append((prompt, g, ans))
            all_r_Ts.append(r_T)

    n_prompts = len(prompt_sizes)
    n_gens = len(all_triples)
    print(f"[plot_vary_T] Loaded {n_prompts} prompts, {n_gens} total generations (empty: {n_empty})")

    if n_prompts == 0:
        raise RuntimeError("No usable samples found.")

    # ========== Phase 2: Score ALL generations ONCE ==========
    print("[plot_vary_T] Scoring all generations.")
    all_rewards = R.score_batch(all_triples)  
    if not isinstance(all_rewards, torch.Tensor):
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    all_rewards = all_rewards.to(device)
    
    r_Ts_t = torch.tensor(all_r_Ts, dtype=torch.float32, device=device)
    prompt_sizes_t = torch.tensor(prompt_sizes, dtype=torch.long, device=device)
    
    prompt_indices = torch.repeat_interleave(
        torch.arange(n_prompts, device=device), 
        prompt_sizes_t
    )
    
    max_rewards_per_prompt = torch.zeros(n_prompts, device=device, dtype=torch.float32)
    max_rewards_per_prompt.fill_(float('-inf'))
    max_rewards_per_prompt.scatter_reduce_(0, prompt_indices, all_rewards, reduce="amax", include_self=True)
    
    if not T_values:
        raise ValueError("T_values must be non-empty")
    print(f"[plot_vary_T] Scoring complete. Now computing deltas for T values: {T_values}")

    # ========== Phase 3: Compute delta for each T ==========
    T_array = torch.tensor(T_values, dtype=torch.float32, device=device)
    delta_means = []
    delta_stds = []
    
    for T_val in T_array.tolist():
        if T_val == 0.0:
            # Hard selection: find argmax reward per prompt
            deltas = torch.zeros(n_prompts, device=device, dtype=torch.float32)
            offset = 0
            for i, k_i in enumerate(prompt_sizes):
                rewards_i = all_rewards[offset:offset+k_i]
                r_Ts_i = r_Ts_t[offset:offset+k_i]
                best_idx = torch.argmax(rewards_i)
                deltas[i] = -r_Ts_i[best_idx]
                offset += k_i
        else:
            scaled = (all_rewards - max_rewards_per_prompt[prompt_indices]) / T_val
            exp_scaled = torch.exp(scaled)
            
            denom = torch.zeros(n_prompts, device=device, dtype=torch.float32)
            denom.scatter_add_(0, prompt_indices, exp_scaled)
            
            weighted = exp_scaled * r_Ts_t
            numer = torch.zeros(n_prompts, device=device, dtype=torch.float32)
            numer.scatter_add_(0, prompt_indices, weighted)
            
            deltas = -numer / denom
        
        delta_mean = deltas.mean().item()
        delta_std = deltas.std().item()
        delta_means.append(delta_mean)
        delta_stds.append(delta_std)
        print(f"  T={T_val:.1f}: delta_mean={delta_mean:.6f}, delta_std={delta_std:.6f}")
    
    T_np = T_array.cpu().numpy()
    delta_np = np.array(delta_means)
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_np, delta_np, color='red', linewidth=2)
    plt.xlabel('$T$', fontsize=14)
    plt.ylabel('$\\delta$', fontsize=14)
    ax = plt.gca()  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delta_vs_T.png', dpi=150, bbox_inches='tight')
    np.savetxt('T_values.txt', T_np)
    np.savetxt('delta_values.txt', delta_np)
    
    print(f"\n[plot_vary_T] Results:")
    print(f"  num_prompts = {n_prompts}")
    print(f"  k mean/min/max = {np.mean(prompt_sizes):.2f} / {min(prompt_sizes)} / {max(prompt_sizes)}")
    print(f"  Saved: delta_vs_T.png, T_values.txt, delta_values.txt")

def main() -> None:
   
    ap = argparse.ArgumentParser(description="Compute δ using lm-eval.")
    
    ap.add_argument("--samples_jsonl", default=None,
                    help="Path/glob pattern to existing samples JSONL file(s)")
    ap.add_argument("--generate_samples", action="store_true",
                    help="Generate samples using lm_eval before computing delta")
    
    ap.add_argument("--k", type=int, default=64,
                    help="Number of samples per prompt")
    ap.add_argument("--model_args", type=str,
                    default="pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9",
                    help="Model arguments for vLLM")
    ap.add_argument("--tasks", type=str, default="gsm8k_cot_self_consistency",
                    help="Task name(s) for lm_eval")
    ap.add_argument("--output_path", type=str, default="out/gsm8k_run",
                    help="Output directory for lm_eval")
    ap.add_argument("--yaml_path", type=str,
                    default="lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml",
                    help="Path to the task YAML configuration file")
    ap.add_argument("--limit", type=float, default=None,
                    help="Limit fraction of dataset to evaluate")
    
    ap.add_argument("--T_values", type=str, default=None, 
                       help="Explicit temperature values (comma-separated), overrides T_min/max/step")
    ap.add_argument("--T_min", type=float, default=0.0,
                    help="Minimum temperature for delta vs T plot")
    ap.add_argument("--T_max", type=float, default=10.0,
                    help="Maximum temperature for delta vs T plot")
    ap.add_argument("--T_step", type=float, default=1.0,
                    help="Temperature step size for delta vs T plot")
    ap.add_argument("--reward", choices=["judge", "http"], default="judge",
                    help="Reward provider type")
    ap.add_argument("--reward_api", default=None,
                    help="URL for HTTP reward API (if --reward http)")
    ap.add_argument("--judge_model", default="mistralai/Mistral-7B-Instruct-v0.3",
                    help="HuggingFace model ID for judge")
    ap.add_argument("--judge_mode", choices=["auto", "gen", "prm"], default="auto",
                    help="Judge mode: auto (detect), gen (generative LM), prm (process reward model)")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Epsilon for judge reward quality tuning (default: 0.0)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for judge noise perturbation (default: 0)")
    
    args = ap.parse_args()
    print(f"[main] Arguments: {vars(args)}")

    if args.T_values:
        T_values = [float(k.strip()) for k in args.T_values.split(",") if k.strip()]
    else:
        if args.T_step <= 0:
            raise ValueError("--T_step must be > 0")
        T_values = np.arange(args.T_min, args.T_max + 0.5 * args.T_step, args.T_step).tolist()
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    if args.generate_samples:
        samples_glob = generate_samples(
            k=args.k,
            model_args=args.model_args,
            tasks=args.tasks,
            output_path=args.output_path,
            yaml_path=args.yaml_path,
            limit=args.limit
        )
    elif args.samples_jsonl:
        samples_glob = args.samples_jsonl
    else:
        raise ValueError("Must specify either --samples_jsonl or --generate_samples")
    
    plot_vary_T(
        T_values, args.k, samples_glob,
        args.reward, args.reward_api, args.judge_model, args.judge_mode, args.epsilon, args.seed
    )

if __name__ == "__main__":
    main()
