Code for "Latent Introspection: Models Can Detect Prior Concept Injections" (TODO: arxiv link)

# Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12+.

```bash
uv sync
```

# Usage

Run all default experiments:

```bash
uv run python main.py
```

By default, this uses `Qwen/Qwen2.5-Coder-32B-Instruct` and `emergent-misalignment/Qwen-Coder-Insecure`, which requires two GPUs. (Tested on 2xH100.) To run on a single GPU without EM experiments:

```bash
uv run python main.py --no-em
```

Results are saved to `outputs/`. Prompts are assembled in main.py, but the cached conversation json for any experiment can be viewed in `prompts/`.

## Selecting experiments

If you want to run just a subset of experiments, or experiments that are not enabled by default, list available experiments using:

```bash
uv run python main.py --list
```

Then, run specific experiment(s):

```bash
uv run python main.py -e top_of_mind_samples -e introspection_yes_shift
```

## Model options

Use a different model (must provide `--model-device`, `--layer-range`, and `--vector-strength` for models not in the built-in settings):

```bash
uv run python main.py --model meta-llama/Llama-3.3-70B-Instruct
```

Override device placement, layer range, or vector strength:

```bash
uv run python main.py --model-device auto --layer-range 21:43 --vector-strength 20.0
```

To run emergent misalignment experiments, supply an `--em-model` and `--em-device`:

```bash
uv run python main.py --em-model emergent-misalignment/Qwen-Coder-Insecure --em-device cuda:1
```

(Without EM settings, emergent misalignment experiments will be automatically disabled.)

## Multi-seed runs

Run experiments across multiple seeds for error bars:

```bash
uv run python main.py --n-seeds 5 --base-seed 42
```

## Prompts

Prompt files are checked for consistency before each run. To regenerate them:

```bash
uv run python main.py --regenerate-prompts
```

To verify they are up-to-date without running experiments:

```bash
uv run python main.py --verify-prompts
```

## Data

- `data/strongreject_dataset.csv` — harmful prompt dataset from [StrongREJECT](https://arxiv.org/abs/2402.10260) (Souly et al., 2024)
- `data/all_truncated_outputs.json` — truncated model output prefixes used for steering vector training diversity, generated using [repeng](https://github.com/vgel/repeng)

## License

MIT. See LICENSE.
