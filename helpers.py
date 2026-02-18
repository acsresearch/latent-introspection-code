import csv
import difflib
import json
import os
import random
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.figure import Figure
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

from repeng import ControlModel, ControlVector, DatasetEntry
from repeng.extract import batched_get_hiddens

ModelType = PreTrainedModel | ControlModel


PROMPTS_DIR = Path("prompts")


class Message(TypedDict):
    role: str
    content: str


class MessageWithContinue(Message, total=False):
    continue_msg: bool  # continue the assistant message (no closing tag)
    add_injection: bool  # marker for where to apply injection at runtime


@dataclass
class Conversation:
    """A single conversation to be used in an experiment."""

    label: str
    yes_token: str
    no_token: str
    extra: dict[str, Any]
    messages: list[MessageWithContinue]


class InjectionDict(TypedDict, total=False):
    cmp_cvec: ControlVector
    cmp_model: PreTrainedModel


@dataclass(frozen=True)
class AnswerFormat:
    """Token handling for yes/no vs 0/1 answers."""

    label: str  # "yes/no" or "0/1"
    yes_token: str  # " yes" or "1"
    no_token: str  # " no" or "0"
    turn_2_format_desc: str  # "The first words of..."
    assistant_prefix: str  # "The answer is" or "The answer is "


@dataclass(frozen=True)
class Intervention:
    """Setup to describe an intervention."""

    label: str
    turn_1_prefix: str
    turn_1_suffix: str
    turn_2_prefix: str
    turn_2_suffix: str


class MessageWithInjection(Message, InjectionDict, total=False):
    continue_msg: bool  # continue the assistant message (no closing tag)


# Type for prompt generator functions
PromptGeneratorFn = Callable[[], Iterator[Conversation]]

# Registry: name -> generator function
PROMPT_GENERATORS: dict[str, PromptGeneratorFn] = {}


def prompts(fn: PromptGeneratorFn) -> PromptGeneratorFn:
    """Decorator to register a prompt generator function."""
    PROMPT_GENERATORS[fn.__name__] = fn
    return fn


class PromptLibrary:
    """Manages loading, saving, and verifying prompt conversations."""

    def __init__(self, base_dir: Path = PROMPTS_DIR):
        self.base_dir = base_dir
        self.conversations: dict[str, list[Conversation]] = {}

    def _group_dir(self, group_name: str) -> Path:
        return self.base_dir / group_name

    def _conv_path(self, group_name: str, label: str) -> Path:
        safe_label = label.replace(" ", "_").replace("/", "_")
        return self._group_dir(group_name) / f"{safe_label}.json"

    def generate_all(self) -> dict[str, list[Conversation]]:
        """Run all registered generators and return conversations by group.

        Raises ValueError if any generator produces duplicate labels.
        """
        result: dict[str, list[Conversation]] = {}
        for name, generator in PROMPT_GENERATORS.items():
            convs = list(generator())
            seen: set[str] = set()
            for conv in convs:
                if conv.label in seen:
                    raise ValueError(
                        f"Duplicate label '{conv.label}' in prompt group '{name}'"
                    )
                seen.add(conv.label)
            result[name] = convs
        return result

    def _order_path(self, group_name: str) -> Path:
        return self._group_dir(group_name) / "_order.json"

    def save(self, group_name: str, conversations: list[Conversation]) -> None:
        """Save conversations to disk, deleting any that were removed."""
        group_dir = self._group_dir(group_name)
        group_dir.mkdir(parents=True, exist_ok=True)

        # Track which files we're writing
        new_paths: set[Path] = set()
        for conv in conversations:
            path = self._conv_path(group_name, conv.label)
            new_paths.add(path)
            with open(path, "w") as f:
                json.dump(asdict(conv), f, indent=2)
                f.write("\n")

        # Save order file
        order_path = self._order_path(group_name)
        with open(order_path, "w") as f:
            json.dump([c.label for c in conversations], f, indent=2)
            f.write("\n")

        # Delete any .json files that aren't in the new set (excluding _order.json)
        for existing in group_dir.glob("*.json"):
            if existing not in new_paths and existing != order_path:
                existing.unlink()

    def load(self, group_name: str) -> list[Conversation]:
        """Load conversations from disk in saved order."""
        group_dir = self._group_dir(group_name)
        if not group_dir.exists():
            return []

        order_path = self._order_path(group_name)
        if not order_path.exists():
            return []

        with open(order_path) as f:
            order = json.load(f)

        conversations = []
        for label in order:
            path = self._conv_path(group_name, label)
            with open(path) as f:
                conversations.append(Conversation(**json.load(f)))
        return conversations

    def load_all(self) -> dict[str, list[Conversation]]:
        """Load all conversation groups from disk."""
        result: dict[str, list[Conversation]] = {}
        if not self.base_dir.exists():
            return result
        for group_dir in sorted(self.base_dir.iterdir()):
            if group_dir.is_dir():
                result[group_dir.name] = self.load(group_dir.name)
        return result

    @staticmethod
    def _word_diff_line(old_line: str, new_line: str) -> str:
        """Create a word-diff style output with color for a single changed line."""
        # ANSI colors
        RED = "\033[31m"
        GREEN = "\033[32m"
        RESET = "\033[0m"
        STRIKETHROUGH = "\033[9m"

        # Use SequenceMatcher for character-level diff
        matcher = difflib.SequenceMatcher(None, old_line, new_line)
        result: list[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                result.append(old_line[i1:i2])
            elif tag == "delete":
                result.append(f"{RED}{STRIKETHROUGH}{old_line[i1:i2]}{RESET}")
            elif tag == "insert":
                result.append(f"{GREEN}{new_line[j1:j2]}{RESET}")
            elif tag == "replace":
                result.append(f"{RED}{STRIKETHROUGH}{old_line[i1:i2]}{RESET}")
                result.append(f"{GREEN}{new_line[j1:j2]}{RESET}")

        return "".join(result)

    def diff(
        self, group_name: str, old: list[Conversation], new: list[Conversation]
    ) -> list[str]:
        """Return unified diff lines with word-level coloring."""
        # ANSI colors
        RED = "\033[31m"
        GREEN = "\033[32m"
        CYAN = "\033[36m"
        RESET = "\033[0m"

        old_by_label = {c.label: c for c in old}
        new_by_label = {c.label: c for c in new}
        all_labels = sorted(set(old_by_label.keys()) | set(new_by_label.keys()))

        diff_lines: list[str] = []

        # Check for order changes
        old_order = [c.label for c in old]
        new_order = [c.label for c in new]
        if old_order != new_order:
            diff_lines.append(f"{CYAN}ORDER CHANGED in {group_name}{RESET}")

        for label in all_labels:
            old_conv = old_by_label.get(label)
            new_conv = new_by_label.get(label)
            path = str(self._conv_path(group_name, label))

            if old_conv is None:
                diff_lines.append(f"{GREEN}+++ NEW: {path}{RESET}")
                continue
            if new_conv is None:
                diff_lines.append(f"{RED}--- DELETED: {path}{RESET}")
                continue

            old_json = json.dumps(asdict(old_conv), indent=2).splitlines()
            new_json = json.dumps(asdict(new_conv), indent=2).splitlines()
            if old_json != new_json:
                diff_lines.append(f"{CYAN}--- {path}{RESET}")
                diff_lines.append(f"{CYAN}+++ {path}{RESET}")

                # Use unified diff to find changed regions, then word-diff them
                for diff_line in difflib.unified_diff(
                    old_json, new_json, lineterm="", n=0
                ):
                    if diff_line.startswith("@@"):
                        diff_lines.append(f"{CYAN}{diff_line}{RESET}")
                    elif diff_line.startswith("---") or diff_line.startswith("+++"):
                        continue  # Skip file headers, we added our own
                    elif diff_line.startswith("-"):
                        # Find corresponding + line for word diff
                        old_content = diff_line[1:]
                        # Look ahead for matching + line (simple heuristic)
                        diff_lines.append(f"{RED}-{old_content}{RESET}")
                    elif diff_line.startswith("+"):
                        new_content = diff_line[1:]
                        diff_lines.append(f"{GREEN}+{new_content}{RESET}")
                    else:
                        diff_lines.append(diff_line)

                # Also show inline word diff for content fields
                old_dict = asdict(old_conv)
                new_dict = asdict(new_conv)
                for key in old_dict:
                    if old_dict[key] != new_dict.get(key):
                        old_val = str(old_dict[key])
                        new_val = str(new_dict.get(key, ""))
                        word_diff = self._word_diff_line(old_val, new_val)
                        diff_lines.append(f"  {CYAN}{key}:{RESET} {word_diff}")

        return diff_lines

    def regenerate(self) -> bool:
        """Regenerate all prompts, print diffs, save to disk. Returns True if any changed."""
        generated = self.generate_all()
        any_changed = False

        for group_name, new_convs in generated.items():
            old_convs = self.load(group_name)
            diff_lines = self.diff(group_name, old_convs, new_convs)

            if diff_lines:
                any_changed = True
                print(f"\n=== {group_name} ===")
                for line in diff_lines:
                    print(line)
                self.save(group_name, new_convs)
                print(
                    f"Saved {len(new_convs)} conversations to {self._group_dir(group_name)}/"
                )
            else:
                print(f"{group_name}: no changes")

        return any_changed

    def verify(self) -> bool:
        """Verify disk matches generated. Returns True if valid, exits if mismatch."""
        generated = self.generate_all()
        mismatches: list[str] = []

        for group_name, new_convs in generated.items():
            old_convs = self.load(group_name)
            diff_lines = self.diff(group_name, old_convs, new_convs)
            if diff_lines:
                mismatches.append(group_name)
                print(f"\n=== MISMATCH: {group_name} ===", file=sys.stderr)
                for line in diff_lines:
                    print(line, file=sys.stderr)

        if mismatches:
            print(
                f"\nPrompt files out of sync: {', '.join(mismatches)}\n"
                "Run with --regenerate-prompts to update.",
                file=sys.stderr,
            )
            sys.exit(1)

        return True


def apply_injection(
    conv: Conversation, injection: InjectionDict | None
) -> list[MessageWithInjection]:
    """Apply injection to a conversation, returning steps for logit_diff_helper etc.

    If injection is None, messages with add_injection are included without injection.
    The add_injection field is always removed from output.
    """
    steps: list[MessageWithInjection] = []
    for msg in conv.messages:
        step: MessageWithInjection = {"role": msg["role"], "content": msg["content"]}
        if msg.get("continue_msg"):
            step["continue_msg"] = True
        if msg.get("add_injection") and injection is not None:
            step = {**step, **injection}
        steps.append(step)
    return steps


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use in filenames."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


@dataclass
class ExperimentLogger:
    """Logger for experiment outputs - saves structured data, plots, and console output."""

    run_dir: Path
    settings: dict[str, Any]
    results: dict[str, Any] = field(default_factory=dict)
    logit_top_k: int = 2048
    _output_file: IO[str] = field(init=False, repr=False)

    @staticmethod
    def make_run_dir(base_dir: str, prefix: str) -> Path:
        """Create a timestamped run directory and return its path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @classmethod
    def create(
        cls,
        settings: dict[str, Any],
        base_dir: str | None = None,
        parent: Path | None = None,
        name: str | None = None,
        logit_top_k: int = 2048,
    ) -> "ExperimentLogger":
        """Create a new logger.

        If parent is provided, creates a subrun directory under it (no timestamp).
        Otherwise, creates a timestamped run directory under base_dir.
        """
        if parent is not None:
            if name is None:
                raise ValueError("name is required when using parent")
            run_dir = parent / name
        else:
            if base_dir is None:
                raise ValueError("base_dir is required when not using parent")
            run_dir = cls.make_run_dir(base_dir, "run")

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "logits").mkdir(exist_ok=True)

        logger = cls(
            run_dir=run_dir, settings=settings, results={}, logit_top_k=logit_top_k
        )
        logger._output_file = open(run_dir / "output.txt", "w")
        return logger

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finalize()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to both stdout and output.txt."""
        print(*args, **kwargs)
        print(*args, **kwargs, file=self._output_file)
        self._output_file.flush()

    def log_experiment(self, name: str, data: dict[str, Any]) -> None:
        """Store structured data for an experiment and save to disk."""
        self.results[name] = data
        self._save_results()

    def _save_results(self) -> None:
        """Write current results to results.json."""
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)

    def save_plot(self, name: str, fig: Figure | None = None, dpi: int = 150) -> str:
        """Save plot and return relative path. Uses current figure if fig is None."""
        rel_path = f"plots/{name}.png"
        if fig is None:
            plt.savefig(self.run_dir / rel_path, dpi=dpi)
        else:
            fig.savefig(self.run_dir / rel_path, dpi=dpi)
        return rel_path

    def save_logits(self, name: str, probs: torch.Tensor) -> str:
        """Save top-k logits as .pt file, return relative path."""
        rel_path = f"logits/{name}.pt"
        k = min(self.logit_top_k, probs.shape[-1])
        top_vals, top_idx = probs.topk(k)
        torch.save({"values": top_vals, "indices": top_idx}, self.run_dir / rel_path)
        return rel_path

    def finalize(self) -> None:
        """Write final results.json and close output file."""
        # Add metadata with settings
        self.results["_meta"] = {
            "model": self.settings["model"],
            "model_device": self.settings["model_device"],
            "em_model": self.settings["em_model"],
            "em_device": self.settings["em_device"],
            "layer_range": self.settings["layer_range"],
            "vector_strength": self.settings["vector_strength"],
            "em_vector_strength": self.settings["em_vector_strength"],
            "seed": self.settings["seed"],
            "n_seeds": self.settings["n_seeds"],
            "timestamp": datetime.now().isoformat(),
        }
        self._save_results()
        self.print(f"Results saved to: {self.run_dir}")
        self._output_file.close()


VECTOR_CACHE_BASE = "outputs/vectors"


def token_id(tokenizer, token: str) -> int:
    return tokenizer.encode(token, add_special_tokens=False)[0]


def load_strongreject_csv(path: str) -> list[str]:
    """Load forbidden prompts from strongreject CSV."""
    prompts = []
    with open(path) as f:
        for row in csv.reader(f, delimiter=",", quotechar='"'):
            if row[-1] != "forbidden_prompt":
                prompts.append(row[-1])
    return prompts


def generation_prompt(tokenizer: PreTrainedTokenizer, persona: str) -> str:
    return cast(
        str,
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"Please talk about {persona}."},
            ],
            add_generation_prompt=True,
            tokenize=False,
        ),
    )


def _model_cache_dir(model_id: str) -> str:
    """Get cache directory for a model, using sanitized model name."""
    safe_name = model_id.replace("/", "_")
    return f"{VECTOR_CACHE_BASE}/{safe_name}"


def train_concept_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    concept: str,
    output_suffixes: list[str],
    model_id: str,
    seed: int = 42,
) -> ControlVector:
    cache_dir = _model_cache_dir(model_id)
    # Include seed in cache path for multi-seed runs
    cache_path = f"{cache_dir}/{concept}_seed{seed}.gguf"
    if os.path.exists(cache_path):
        return ControlVector.import_gguf(cache_path)

    # Seed controls sklearn's randomized PCA (used for large matrices).
    # sklearn's PCA uses randomized SVD solver for large data and respects numpy's
    # random state. repeng doesn't expose random_state param yet.
    np.random.seed(seed)
    dataset: list[DatasetEntry] = []
    persona_prompt = generation_prompt(tokenizer, concept)
    default_prompt = generation_prompt(tokenizer, "anything")
    for suffix in output_suffixes:
        dataset.append(
            DatasetEntry(
                positive=persona_prompt + suffix,
                negative=default_prompt + suffix,
            )
        )
    vector = ControlVector.train(
        model,
        tokenizer,
        dataset,
        method="pca_center",
        batch_size=64,
    )
    os.makedirs(cache_dir, exist_ok=True)
    vector.export_gguf(cache_path)
    return vector


def train_model_contrastive_vector(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    name: str,
    model_id: str,
    seed: int = 42,
) -> ControlVector:
    """Helper to train a cvec with one prompt, two models instead of two prompts, one model"""
    cache_dir = _model_cache_dir(model_id)
    # Include seed in cache path for multi-seed runs
    cache_path = f"{cache_dir}/{name}_seed{seed}.gguf"
    if os.path.exists(cache_path):
        return ControlVector.import_gguf(cache_path)

    # Seed controls sklearn's randomized PCA - see comment in train_concept_vector
    np.random.seed(seed)

    def compute_model_contrastive_hiddens(
        train_strs: list[str],
        hidden_layers: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> dict[int, npt.NDArray[np.floating[Any]]]:
        a_train_strs, b_train_strs = train_strs[::2], train_strs[1::2]
        assert len(a_train_strs) == len(b_train_strs)

        a_hiddens = batched_get_hiddens(
            model_a, tokenizer, a_train_strs, hidden_layers, batch_size
        )
        b_hiddens = batched_get_hiddens(
            model_b, tokenizer, b_train_strs, hidden_layers, batch_size
        )
        interleaved: dict[int, npt.NDArray[np.floating[Any]]] = {}
        for layer in hidden_layers:
            ah, bh = a_hiddens[layer], b_hiddens[layer]
            i = np.stack((ah, bh))
            i = i.transpose(1, 0, *range(2, i.ndim))
            i = i.reshape((ah.shape[0] + bh.shape[0], ah.shape[1]))  # ex*2, hidden_dim
            interleaved[layer] = i
        return interleaved

    vector = ControlVector.train(
        model=model_a,
        tokenizer=tokenizer,
        dataset=[DatasetEntry(positive=x, negative=x) for x in prompts],
        compute_hiddens=compute_model_contrastive_hiddens,
        method="pca_center",
    )
    os.makedirs(cache_dir, exist_ok=True)
    vector.export_gguf(cache_path)
    return vector


def train_vectors(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    em_model: PreTrainedModel | None,
    concepts: list[str],
    output_suffixes: list[str],
    strongreject: list[str],
    model_id: str,
    seed: int = 42,
) -> dict[str, ControlVector]:
    print(f"Training steering vectors (seed={seed}):")
    vectors: dict[str, ControlVector] = {}
    for concept in concepts:
        print(f"  {concept}...")
        vectors[concept] = train_concept_vector(
            model, tokenizer, concept, output_suffixes, model_id, seed=seed
        )

    if em_model is not None:
        print("  em (contrastive)...")
        vectors["em"] = train_model_contrastive_vector(
            em_model, model, tokenizer, strongreject, "em", model_id, seed=seed
        )
    return vectors


def _is_multi_gpu_model(model: ModelType) -> bool:
    """Check if model uses multiple GPUs via device_map."""
    base_model = model.model if isinstance(model, ControlModel) else model
    device_map: dict[str, Any] | None = getattr(base_model, "hf_device_map", None)
    return device_map is not None and len(set(device_map.values())) > 1


def move_cache_to_device(cache: DynamicCache, device: torch.device) -> None:
    for layer_ in cache.layers:  # type: ignore[attr-defined]
        layer: Any = layer_
        if layer.is_initialized:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
            layer.device = device
            # For sliding window layers
            if hasattr(layer, "_sliding_window_tensor"):
                layer._sliding_window_tensor = layer._sliding_window_tensor.to(device)


def prefill(
    kv: DynamicCache | None, model: ModelType, tokens: torch.Tensor
) -> torch.Tensor:
    # Skip cache movement for multi-GPU models - transformers handles device placement
    if kv is not None and not _is_multi_gpu_model(model):
        move_cache_to_device(kv, model.device)
    with torch.inference_mode():
        return (
            model(
                input_ids=tokens.to(model.device),
                past_key_values=kv,
                use_cache=kv is not None,
            )
            .logits[:, -1]
            .cpu()
        )


def logit(p: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def generate_orderings(n_concepts: int, n_orderings: int, seed: int) -> list[list[int]]:
    """
    Generate random permutations for label ordering.

    Returns list of permutations, where perm[shuffled_idx] = canonical_idx.
    E.g., perm = [2, 0, 1] means:
      - shuffled position 0 shows canonical concept 2
      - shuffled position 1 shows canonical concept 0
      - shuffled position 2 shows canonical concept 1
    """
    rng = random.Random(seed)
    orderings = []
    for _ in range(n_orderings):
        perm = list(range(n_concepts))
        rng.shuffle(perm)
        orderings.append(perm)
    return orderings


def map_to_canonical(
    shuffled_probs: npt.NDArray[np.floating[Any]],
    perm: list[int],
) -> npt.NDArray[np.floating[Any]]:
    """
    Map probabilities from shuffled order back to canonical order.

    Args:
        shuffled_probs: Array of shape [n_concepts] with probs for shuffled labels 0,1,2,...
        perm: Permutation where perm[shuffled_idx] = canonical_idx

    Returns:
        canonical_probs: Array of shape [n_concepts] with probs in canonical order
    """
    n = len(perm)
    canonical_probs = np.zeros(n, dtype=shuffled_probs.dtype)
    for shuffled_idx, canonical_idx in enumerate(perm):
        canonical_probs[canonical_idx] = shuffled_probs[shuffled_idx]
    return canonical_probs


def average_confusion_matrices(
    matrices: list[npt.NDArray[np.floating[Any]]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Average multiple confusion matrices.

    Args:
        matrices: List of confusion matrices, each of shape [n_layers, n_concepts, n_concepts]

    Returns:
        Averaged confusion matrix of same shape
    """
    stacked = np.stack(matrices, axis=0)
    return stacked.mean(axis=0)


def logit_diff_helper(
    tokenizer: PreTrainedTokenizer,
    steps: list[MessageWithInjection],
    model: PreTrainedModel,
    steering_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Check that apply_injection was called (add_injection should be removed)
    for step in steps:
        if step.get("add_injection"):  # type: ignore[typeddict-item]
            raise ValueError(
                "Step contains 'add_injection' - did you forget to call apply_injection()?"
            )
    base_kv, expr_kv = DynamicCache(), DynamicCache()
    base_probs, expr_probs = None, None
    n_tokens = 0
    for i in range(len(steps)):
        if steps[i]["role"] == "system":
            continue

        tokens = cast(
            torch.Tensor,
            tokenizer.apply_chat_template(
                [
                    {"role": steps[j]["role"], "content": steps[j]["content"]}
                    for j in range(i + 1)
                ],
                continue_final_message=steps[i].get("continue_msg", False),
                return_tensors="pt",
            ),
        )[:, n_tokens:]
        n_tokens += tokens.shape[1]

        base_probs = prefill(base_kv, model, tokens).softmax(dim=-1).squeeze()
        if cvec := steps[i].get("cmp_cvec"):
            wrapped = ControlModel(model, list(steering_layers))
            wrapped.set_control(cvec)
            expr_probs = prefill(expr_kv, wrapped, tokens).softmax(dim=-1).squeeze()
            wrapped.reset()
            wrapped.unwrap()
        elif cmp_model := steps[i].get("cmp_model"):
            expr_probs = prefill(expr_kv, cmp_model, tokens).softmax(dim=-1).squeeze()
        else:
            expr_probs = prefill(expr_kv, model, tokens).softmax(dim=-1).squeeze()

    assert base_probs is not None and expr_probs is not None
    return base_probs, expr_probs


def prefill_hidden_states(
    kv: DynamicCache | None, model: ModelType, tokens: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    # Skip cache movement for multi-GPU models - transformers handles device placement
    if kv is not None and not _is_multi_gpu_model(model):
        move_cache_to_device(kv, model.device)
    with torch.inference_mode():
        return model(
            input_ids=tokens.to(model.device),
            past_key_values=kv,
            use_cache=kv is not None,
            output_hidden_states=True,
        ).hidden_states


def logit_lens_helper(
    tokenizer: PreTrainedTokenizer,
    steps: list[MessageWithInjection],
    model: PreTrainedModel,
    steering_layers: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Returns (base_probs, expr_probs) where each is a list of prob tensors per layer."""
    # Check that apply_injection was called (add_injection should be removed)
    for step in steps:
        if step.get("add_injection"):  # type: ignore[typeddict-item]
            raise ValueError(
                "Step contains 'add_injection' - did you forget to call apply_injection()?"
            )
    base_kv, expr_kv = DynamicCache(), DynamicCache()
    base_hs, expr_hs = None, None
    n_tokens = 0
    for i in range(len(steps)):
        if steps[i]["role"] == "system":
            continue

        tokens = cast(
            torch.Tensor,
            tokenizer.apply_chat_template(
                [
                    {"role": steps[j]["role"], "content": steps[j]["content"]}
                    for j in range(i + 1)
                ],
                continue_final_message=steps[i].get("continue_msg", False),
                return_tensors="pt",
            ),
        )[:, n_tokens:]
        n_tokens += tokens.shape[1]

        base_hs = prefill_hidden_states(base_kv, model, tokens)
        if cvec := steps[i].get("cmp_cvec"):
            wrapped = ControlModel(model, list(steering_layers))
            wrapped.set_control(cvec)
            expr_hs = prefill_hidden_states(expr_kv, wrapped, tokens)
            wrapped.reset()
            wrapped.unwrap()
        elif cmp_model := steps[i].get("cmp_model"):
            expr_hs = prefill_hidden_states(expr_kv, cmp_model, tokens)
        else:
            expr_hs = prefill_hidden_states(expr_kv, model, tokens)

    assert base_hs is not None and expr_hs is not None
    base_ll: list[torch.Tensor] = []
    expr_ll: list[torch.Tensor] = []
    for layer_idx, (base_h, expr_h) in enumerate(zip(base_hs, expr_hs)):
        with torch.inference_mode():
            # last layer hidden is already normed, don't double-norm
            if layer_idx == len(base_hs) - 1:
                base_normed = base_h[0, -1, :]
                expr_normed = expr_h[0, -1, :]
            else:
                base_normed = model.model.norm(base_h[0, -1, :])
                expr_normed = model.model.norm(expr_h[0, -1, :])

            base_probs = model.lm_head(base_normed).softmax(dim=-1).cpu()
            expr_probs = model.lm_head(expr_normed).softmax(dim=-1).cpu()

        base_ll.append(base_probs)
        expr_ll.append(expr_probs)

    return base_ll, expr_ll


def build_injections(
    vectors: dict[str, ControlVector],
    em_model: PreTrainedModel | None,
    concepts: list[str],
    vector_strength: float,
    em_vector_strength: float | None,
) -> list[tuple[str, InjectionDict]]:
    """Build list of (label, injection_dict) for all concepts + EM variants."""
    injections: list[tuple[str, InjectionDict]] = [
        (concept, {"cmp_cvec": vector_strength * vectors[concept]})
        for concept in concepts
    ]
    # Only add EM injections if EM model is available
    if em_model is not None:
        assert em_vector_strength is not None, (
            "em_vector_strength required when em_model is set"
        )
        injections.append(("em-vec", {"cmp_cvec": em_vector_strength * vectors["em"]}))
        injections.append(("em-ft", {"cmp_model": em_model}))
    return injections
