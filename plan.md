# wscribe Development Plan

Two workstreams: add an `mlx-whisper` backend optimised for Apple Silicon, and migrate the project tooling from Poetry/pip to `uv` for a simpler end-user install experience on Mac.

## Commits

### [x] 1. `docs: add plan.md`
Write this file and commit it.

### [x] 2. `build: migrate pyproject.toml from Poetry to uv/hatchling`
- Replace `[tool.poetry]` with PEP 621 `[project]`
- Replace `poetry-core` build backend with `hatchling`
- Add `[tool.hatch.build.targets.wheel] packages = ["src/wscribe"]` for src/ layout
- Convert `^X.Y.Z` constraints to `>=X.Y.Z`
- Replace `[tool.poetry.group.*]` with PEP 735 `[dependency-groups]`
- Add `[project.optional-dependencies] mlx = ["mlx-whisper>=0.4.3"]`
- Add `[[tool.uv.index]]` for test-PyPI
- Delete `poetry.lock`; generate `uv.lock` via `uv lock`

### [x] 3. `build: migrate Makefile from Poetry to uv`
- Replace all `poetry` calls with `uv` equivalents
- `poetry install --sync` → `uv sync --all-groups`
- `poetry show` → `uv tree`
- `poetry build` → `uv build`
- `poetry publish` → `uv publish`
- `poetry publish -r test-pypi` → `uv publish --index testpypi`
- `poetry run pytest` → `uv run pytest`
- `poetry run mypy .` → `uv run mypy .`
- `poetry version patch` → `uv version --bump patch`
- `poetry version prerelease` → `uv version --bump patch --bump alpha`

### [x] 4. `feat: add MLXWhisperBackend for Apple Silicon`
- Create `src/wscribe/backends/mlxwhisper.py`
- Override `supported_model_sizes()` with mlx-community HF repo list
- `model_path()` returns `self.model_size` directly (it is the HF repo string)
- `load()` is a no-op (mlx-whisper auto-loads/caches internally)
- `transcribe()` calls `mlx_whisper.transcribe()` and maps output to `list[TranscribedData]`
- Wrap `import mlx_whisper` in try/except for a clear error on non-Apple hardware

### [x] 5. `refactor: move decode_audio import into convert_audio()`
- In `src/wscribe/core.py`, move `from faster_whisper.audio import decode_audio` from module level into the `convert_audio()` method body
- This prevents importing `faster-whisper` just by importing `core.py` when using the mlx backend

### [x] 6. `feat: add --backend option to transcribe CLI command`
- Add `click.Choice(["faster-whisper", "mlx-whisper"])` option `--backend / -b` defaulting to `"faster-whisper"`
- When `mlx-whisper`: instantiate `MLXWhisperBackend`; warn if `--gpu` also passed (mlx always uses Apple Silicon GPU)
- When `mlx-whisper`: accept free-string `--model` (not restricted to `SUPPORTED_MODELS`)

### [x] 7. `docs: update README install instructions for uv`
- Update `README.org` installation section with uv-based install
- Update `docs/README.md` to mirror
- End-user flow: `brew install ffmpeg`, then `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `uv tool install wscribe` (or `uv tool install 'wscribe[mlx]'` for Apple Silicon)
- Keep FFmpeg as an explicit prerequisite

### [x] 8. `docs: update research.md with mlx backend and uv sections`
- Add mlx-whisper backend to §5 Backends
- Add uv tooling details to §10 Configuration & Packaging
- Update §13 Extension Points with the new backend as a worked example
