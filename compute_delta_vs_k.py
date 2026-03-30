import argparse  # For parsing command-line arguments
import sys
import glob  # For file pattern matching
import json  # For working with JSON data
import re  # Regular expressions for parsing strings
import subprocess  # For running external commands
from typing import List, Tuple, Dict, Iterable, Optional  # For type hinting
from functools import lru_cache  # Cache string parsing helpers

import numpy as np
import torch
from matplotlib import pyplot as plt

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, logging
logging.set_verbosity_error()  # Suppress warnings from transformers library unless there is an error

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


n_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpus}")

# Enable TF32 for H100 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Parsing the answer
RE_THE_ANSWER_IS=re.compile(r"The answer is (\-?[0-9\.\,]*[0-9]+)")
RE_HASHES=re.compile(r"####\s*(-?\d[\d,\.]*)")
RE_GENERIC_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")

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


class JudgeReward:
    """
    Supports two judge backends:
      1) A AutoModelForCausalLM that produces a scalar score.
      2) A AutoModel process reward model that scores <extra_0> step separators.
    """

    def __init__(self, model_name: str, max_length: int = 1024, mode: str = "auto", epsilon: float = 0.0, seed: Optional[int] = None):
        self.model_name = model_name
        self.max_length = max_length

        mode = (mode or "auto").lower()
        if mode not in {"auto", "gen", "prm"}:
            raise ValueError("--judge_mode must be one of: auto, gen, prm")

        if mode == "auto":
            self.mode = "prm" if re.search(r"\bprm\b", model_name, flags=re.IGNORECASE) else "gen"
        else:
            self.mode = mode

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        device_map = "balanced" if n_gpus > 1 else "auto"
        self._model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
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
                print("[JudgeReward] Using flash attention for generative judge")
            except Exception as e:
                print(f"[JudgeReward] Could not use flash attention for generative judge: {e}")
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
            self._device = next(self.model.parameters()).device

            self._num = re.compile(r"(-?\d+(?:\.\d+)?)")
            # self.prompt_template = (
            #     "Score the answer of the student based on its logical consistency of reasoning "
            #     "and accuracy of the final answer. The output should be a number between 0 to 10.\n"
            #     "[question]{q}\n[reasoning of the student]{sol}\n[final answer of the student]{ans}\n[Score]:"
            # )
            self.prompt_template = "Score 0-10:\n[Q]{q}\n[A]{sol}\n[Final]{ans}\nScore:"
        self.epsilon = epsilon # set to default value of 0.0 for backward compatibility; used to tune judge quality
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def _fmt_gen(self, q: str, sol: str, ans: str) -> str:
        return self.prompt_template.format(
            q=(q or "")[:400].strip(),
            sol=(sol or "")[:600].strip(),
            ans=(ans or "")[:100].strip(),
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
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _parse_batch(self, texts: List[str]) -> List[float]:
        """
        Parses texts to extract scores between 0 and 10.
        """
        scores = []
        for text in texts:
            m = self._num.search(text or "")
            v = float(m.group(1)) if m else 0.0
            v = v + 10 * self.epsilon * self.rng.standard_normal()
            scores.append(max(0,min(v,10))) # Clamp score between 0 and 10
        return scores
    
    def score_batch(self, triples: List[Tuple[str, str, str]], batch_size: int = 256) -> torch.Tensor:
        """Scores a batch of (question, solution, answer) triples.
        """
        if not triples:
            return torch.empty((0,), dtype=torch.float32, device=self._device)

        if self.mode == "prm":
            return self._score_batch_prm(triples, batch_size=batch_size)
        return self._score_batch_gen(triples, batch_size=batch_size)

    def _score_batch_gen(self, triples: List[Tuple[str, str, str]], batch_size: int) -> torch.Tensor:
        all_scores: List[float] = []
        n_batches = (len(triples) + batch_size - 1) // batch_size
        print(f"[score_batch:gen] Scoring {len(triples)} samples in {n_batches} batches of size {batch_size}")

        for batch_idx in range(0, len(triples), batch_size):
            batch_triples = triples[batch_idx : batch_idx + batch_size]
            batch_texts = [self._fmt_gen(q, sol, ans) for q, sol, ans in batch_triples]
            with torch.inference_mode():
                enc = self.tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self._device, non_blocking=True) for k, v in enc.items()}
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=4,
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

            if (batch_idx // batch_size + 1) % 20 == 0:
                print(f"  Batch {batch_idx // batch_size + 1}/{n_batches}")

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
                print('PRM scores:', scores.detach().cpu().tolist())
                all_scores.extend(scores.detach().cpu().tolist())

            if (batch_idx // batch_size + 1) % 20 == 0:
                print(f"  Batch {batch_idx // batch_size + 1}/{n_batches}")

        return torch.tensor(all_scores, dtype=torch.float32, device=self._device)

def modify_yaml_repeats(yaml_path:str, k:int)->None:
    """
    modify the reapts parameter in YAML task configuration.
    """
    with open(yaml_path,'r') as f:
        content = f.read()
    new_content=re.sub(r"repeats: \s*\d+", f"repeats: {k}", content)
    with open(yaml_path, 'w') as f:
        f.write(new_content)
    print(f"[modify_yaml_repeats] Modified {yaml_path} to have repeats={k}")

def run_lm_eval(model_args: str, tasks: str, output_path: str, limit: Optional[float] = None)-> None:
    """
    Runs the lm_eval command with the specified parameters to generate inference time data.
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
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print(f"[run_lm_eval] Running: {' '.join(cmd)}") # joins a list of strings into a single string, with spaces between each element.
    result = subprocess.run(cmd, capture_output=False, text=True) # running subprocess on the terminal, capture_output=False means output will be printed to console directly
    if result.returncode != 0: # if the command failed
        raise RuntimeError(f"lm_eval failed with return code {result.returncode}")


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


def compute_delta_for_k(samples_glob: str, T: float, judge: JudgeReward, device: torch.device) -> Tuple[float, float, int]:
    all_triples = []
    all_r_Ts = []
    prompt_sizes = []
    
    for rec in iter_jsonl(samples_glob):
        doc = rec.get("doc") or {}
        prompt = doc.get("question") or rec.get("prompt") or rec.get("inputs") or "" # get the question/prompt
        gold = doc.get("answer") or rec.get("target") or "" # get the gold answer
        
        gens = [] # generated answers
        if "resps" in rec:
            rs = rec["resps"]
            if isinstance(rs, list):
                if rs and isinstance(rs[0], list):
                    gens.extend([s for s in rs[0] if isinstance(s, str)])
                else:
                    gens.extend([s for s in rs if isinstance(s, str)])
        
        if not gens:
            continue
        
        prompt_sizes.append(len(gens)) # say [3, 2]
        for g in gens:
            ans = extract_num(g) or "" # generated final answer
            r_T = exact_numeric_equal(g, gold) # reward based on exact numeric equality g=generated, gold=actual answer
            all_triples.append((prompt, g, ans))
            all_r_Ts.append(r_T)
    
    n_prompts = len(prompt_sizes) # n_prompts = 2 (two questions)
    if n_prompts == 0:
        return 0.0, 0.0, 0
    
    print(f"[compute_delta_for_k] {n_prompts} prompts, {len(all_triples)} generations, k≈{sum(prompt_sizes)/n_prompts:.1f}")
    
    # Score all generations
    all_rewards = judge.score_batch(all_triples).to(device) # [7, 5, 8, 6, 9]
    r_Ts_t = torch.tensor(all_r_Ts, dtype=torch.float32, device=device) # [1, 0, 1, 0, 1] 
    prompt_sizes_t = torch.tensor(prompt_sizes, dtype=torch.long, device=device)
    
    # Create prompt indices
    prompt_indices = torch.repeat_interleave(
        torch.arange(n_prompts, device=device), prompt_sizes_t
    ) # [0, 0, 0, 1, 1] 
    
    # Compute delta with softmax weighting
    if T == 0.0:
        # Hard argmax
        deltas = torch.zeros(n_prompts, device=device)
        offset = 0
        for i, k_i in enumerate(prompt_sizes):
            rewards_i = all_rewards[offset:offset+k_i]
            r_Ts_i = r_Ts_t[offset:offset+k_i]
            best_idx = torch.argmax(rewards_i)
            deltas[i] = -r_Ts_i[best_idx]
            offset += k_i
    else:
        # Softmax weighting
        max_rewards = torch.zeros(n_prompts, device=device).fill_(float('-inf')) # [-inf, -inf]
        max_rewards.scatter_reduce_(0, prompt_indices, all_rewards, reduce="amax", include_self=True) # [8, 9]
        
        scaled = (all_rewards - max_rewards[prompt_indices]) / T # [-1/T, -3/T, 0/T, -3/T, 0/T]
        exp_scaled = torch.exp(scaled)
        
        denom = torch.zeros(n_prompts, device=device)
        denom.scatter_add_(0, prompt_indices, exp_scaled) # [sum_prompt0, sum_prompt1]
        
        weighted = exp_scaled * r_Ts_t
        numer = torch.zeros(n_prompts, device=device)
        numer.scatter_add_(0, prompt_indices, weighted)
        
        deltas = -numer / denom
    
    return deltas.mean().item(), deltas.std().item(), n_prompts

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute delta vs k for fixed T")
    
    parser.add_argument("--T", type=float, required=True, help="Temperature")
    parser.add_argument("--k_min", type=int, default=4, help="Minimum number of inferences per prompt")
    parser.add_argument("--k_max", type=int, default=64, help="Maximum number of inferences per prompt")
    parser.add_argument("--k_step", type=int, default=4, help="Step size for k values")
    parser.add_argument("--k_values", type=str, default=None, 
                       help="Explicit inferences per prompt values (comma-separated), overrides k_min/max/step")
    
    parser.add_argument("--model_args", type=str, 
                       default="pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9")
    parser.add_argument("--tasks", type=str, default="gsm8k_cot_self_consistency")
    parser.add_argument("--output_path", type=str, default="out/gsm8k_k_sweep")
    parser.add_argument("--limit", type=float, default=None, help="Limit fraction of dataset")
    parser.add_argument("--yaml_path", type=str, 
                       default="lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml")
    
    parser.add_argument("--skip_lm_eval", action="store_true", 
                       help="Skip running lm_eval, use existing samples")
    parser.add_argument("--samples_pattern", type=str, default=None,
                       help="Pattern for samples files, use {k} as placeholder for k value")
    
    parser.add_argument("--judge_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    
    parser.add_argument("--plot_output", type=str, default="delta_vs_k.png")
    
    parser.add_argument("--epsilon", type=float, default=0.0,
                    help="Epsilon for judge reward quality tuning (default: 0.0)")
    
    parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for judge noise perturbation (default: 0)")
    
    args = parser.parse_args()
    
    if args.k_values:
        k_values = [int(k) for k in args.k_values.split(",")]
    else:
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    
    print(f"[Main] k values to evaluate: {k_values}")
    print(f"[Main] T = {args.T}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    # ========== Phase 1: Run all lm_eval jobs ==========
    samples_globs = {}
    
    if not args.skip_lm_eval:
        print("\n" + "="*60)
        print("[Main] Phase 1: Running lm_eval for all k values")
        print("="*60)
        
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"[Main] Running lm_eval for k = {k}")
            print(f"{'='*60}")
            
            # Modify YAML
            modify_yaml_repeats(args.yaml_path, k)
            
            # Run lm_eval
            k_output_path = f"{args.output_path}_k{k}"
            run_lm_eval(args.model_args, args.tasks, k_output_path, args.limit)
            
            # Store samples glob pattern
            samples_globs[k] = f"{k_output_path}/**/samples_*.jsonl"
        
        print("\n" + "="*60)
        print("[Main] Phase 1 complete. All lm_eval runs finished.")
        print("="*60)
    else:
        # Pre-generated samples
        for k in k_values:
            if args.samples_pattern:
                samples_globs[k] = args.samples_pattern.format(k=k)
            else:
                samples_globs[k] = f"{args.output_path}_k{k}/**/samples_*.jsonl"
    
    # ========== Phase 2: Load judge model and compute delta ==========
    print("\n" + "="*60)
    print("[Main] Phase 2: Loading judge model and computing delta")
    print("="*60)
    
    judge = JudgeReward(model_name=args.judge_model, epsilon=args.epsilon, seed=args.seed)
    
    results = {"k": [], "delta_mean": [], "delta_std": [], "n_prompts": []}
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"[Main] Computing delta for k = {k}")
        print(f"{'='*60}")
        
        samples_glob = samples_globs[k]
        
        # Compute delta
        delta_mean, delta_std, n_prompts = compute_delta_for_k(samples_glob, args.T, judge, device)
        
        results["k"].append(k)
        results["delta_mean"].append(delta_mean)
        results["delta_std"].append(delta_std)
        results["n_prompts"].append(n_prompts)
        
        print(f"[Main] k={k}: delta_mean={delta_mean:.6f}, delta_std={delta_std:.6f}, n={n_prompts}")
    
    # Plot results
    k_arr = np.array(results["k"])
    delta_arr = np.array(results["delta_mean"])
    delta_std_arr = np.array(results["delta_std"])
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_arr, delta_arr, 
                 color='red', linewidth=2, marker='o', markersize=8, capsize=4)
    plt.xlabel('$k$', fontsize=14)
    plt.ylabel('$\\delta$', fontsize=14)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_output, dpi=150, bbox_inches='tight')
    print(f"\n[Main] Saved plot to {args.plot_output}")
    
    # Print summary
    print(f"\n[Main] Summary:")
    print(f"  k values: {k_arr.tolist()}")
    print(f"  delta means: {delta_arr.tolist()}")

if __name__ == "__main__":
    main()
