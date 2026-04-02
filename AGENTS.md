# Repository Guidelines

## Project Structure & Module Organization
This repository is intentionally small. [`train.py`](/home/mathias/dev/autoresearch/train.py) is the main research surface: model definition, optimizer logic, and the 5-minute training loop all live there. [`prepare.py`](/home/mathias/dev/autoresearch/prepare.py) handles one-time dataset download, tokenizer training, dataloading, and evaluation utilities; treat its constants as stable unless a change is clearly intentional. [`program.md`](/home/mathias/dev/autoresearch/program.md) contains the baseline agent instructions. Supporting files are [`README.md`](/home/mathias/dev/autoresearch/README.md), [`pyproject.toml`](/home/mathias/dev/autoresearch/pyproject.toml), the lockfile, and light artifacts such as [`progress.png`](/home/mathias/dev/autoresearch/progress.png) and [`analysis.ipynb`](/home/mathias/dev/autoresearch/analysis.ipynb).

## Build, Test, and Development Commands
Use `uv` for all local work:

- `uv sync` installs the pinned Python dependencies.
- `uv run prepare.py` downloads shards into `~/.cache/autoresearch/` and trains the tokenizer.
- `uv run prepare.py --num-shards 8` performs a smaller setup pass for quick validation.
- `uv run train.py` launches a single benchmark training run and reports `val_bpb`.

Set runtime overrides with env vars when needed, for example `AUTORESEARCH_DEVICE_INDEX=0 uv run train.py`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for tunable constants and env-var names, and short docstrings where behavior is non-obvious. Keep changes localized and readable; this project favors a single-file training script over extra abstraction. There is no configured formatter or linter in `pyproject.toml`, so match the surrounding style and keep imports and print diagnostics tidy.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes by running the smallest realistic workflow first (`uv run prepare.py --num-shards 8` if setup is needed), then `uv run train.py`. For training changes, include the observed hardware, whether the run used CUDA or ROCm, and the resulting `val_bpb`. If you touch data loading or tokenizer logic, confirm the cache paths under `~/.cache/autoresearch/` are still populated as expected.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Improve ROCm runtime reporting` and scoped fixes like `fix(train): make NaN fast-fail check explicit`. Prefer concise subjects under about 72 characters. PRs should explain the hypothesis, summarize code changes, and include the command(s) run plus before/after training results when behavior or performance changes. Link related issues or discussions, and attach logs or screenshots only when they clarify a regression or hardware-specific result.
