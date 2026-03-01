# wscribe — Internal Developer Reference

> **Version:** 0.1.5  
> **Author:** Hrishikesh Barman `<oss@geekodour.org>`  
> **License:** MIT  
> **Homepage:** https://github.com/geekodour/wscribe  
> **Companion app:** https://github.com/geekodour/wscribe-editor

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Model](#3-data-model)
4. [Abstract Base Classes](#4-abstract-base-classes)
5. [Backends](#5-backends)
6. [Sources](#6-sources)
7. [Writers](#7-writers)
8. [CLI](#8-cli)
9. [Logging](#9-logging)
10. [Configuration & Packaging](#10-configuration--packaging)
11. [Tooling](#11-tooling)
12. [Tests](#12-tests)
13. [Design Patterns & Extension Points](#13-design-patterns--extension-points)

---

## 1. Project Overview

`wscribe` is a CLI transcription tool built on top of [faster-whisper](https://github.com/guillaumekln/faster-whisper) (a CTranslate2-optimised implementation of OpenAI Whisper). It was created at [sochara.org](https://www.sochara.org/) to process large volumes of audio recordings that need to be transcribed and archived.

**Key characteristics:**

- Accepts any audio/video file that FFmpeg can decode (all common containers and codecs).
- Produces **word-level timestamps and per-word confidence scores** — this is the primary differentiator from subtitle-only tooling.
- Runs on CPU (int8 quantisation) or CUDA GPU (float16 quantisation).
- Optionally applies VAD (Voice Activity Detection) filtering via faster-whisper's built-in support.
- Exports to JSON (word-level), SRT, or WebVTT.
- The JSON output is designed to feed the [wscribe-editor](https://github.com/geekodour/wscribe-editor) web app for manual correction.

**Hard external dependencies (not managed by pip):**

| Dependency | Role |
|---|---|
| FFmpeg | Audio/video decoding — used internally by `faster_whisper.audio.decode_audio` |
| `WSCRIBE_MODELS_DIR` env var | Must be set; points to the directory containing downloaded model subdirectories |

---

## 2. Repository Layout

```
wscribe/
├── src/
│   └── wscribe/
│       ├── __init__.py             # Logging bootstrap (structlog + stdlib logging)
│       ├── core.py                 # TypedDicts, SUPPORTED_MODELS, Backend and Audio ABCs
│       ├── writers.py              # Output format writers (JSON, SRT, VTT) + WRITERS registry
│       ├── backends/
│       │   └── fasterwhisper.py    # FasterWhisperBackend — the only implemented backend
│       ├── cli/
│       │   └── main.py             # Click CLI: `transcribe` and `info` commands
│       └── sources/
│           └── local.py            # LocalAudio — the only implemented source
├── tests/
│   └── test_wscribe.py             # Integration test for inference pipeline
├── examples/
│   ├── assets/
│   │   └── jfk.wav                 # Sample audio used in tests (JFK speech excerpt)
│   └── output/
│       ├── sample.json             # Example JSON output (empty in repo — placeholder)
│       ├── sample.srt              # Example SRT output (713 lines)
│       └── sample.vtt              # Example WebVTT output (537 lines)
├── scripts/
│   ├── fw_dw_hf_wo_lfs.sh          # Downloads a faster-whisper model from HuggingFace
│   └── speed_check.sh              # Benchmarks all 4 models × 2 devices (8 runs total)
├── docs/
│   └── README.md                   # Markdown version of README.org (used as PyPI readme)
├── README.org                      # Primary documentation in Org-mode
├── pyproject.toml                  # Poetry package manifest
├── Makefile                        # Project task runner
└── Makefile.common                 # Shared Makefile utilities (auto-help, template checks)
```

---

## 3. Data Model

**File:** `src/wscribe/core.py`

### `SUPPORTED_MODELS`

```python
SUPPORTED_MODELS = ["tiny", "small", "medium", "large-v2"]
```

A plain `list[str]` used both by `Backend.__post_init__` for validation and by the CLI's `--model` `click.Choice`.

---

### `WordData` (TypedDict)

Represents a single transcribed word.

| Field | Type | Notes |
|---|---|---|
| `text` | `str` | The word string, including surrounding whitespace as returned by faster-whisper. Stripped to bare text by `WriteJSON.transform_result`. |
| `start` | `float \| str` | Seconds from audio start (float) as produced by the backend. Converted to `"HH:MM:SS.mmm"` string in-place by `WriteJSON`. |
| `end` | `float \| str` | Same as `start`. |
| `score` | `float` | Word-level confidence from faster-whisper (`w.probability`), rounded to 2 decimal places. Range 0.0–1.0. |

---

### `TranscribedData` (TypedDict)

Represents a single transcribed segment (a contiguous speech run, roughly sentence-length).

| Field | Type | Notes |
|---|---|---|
| `text` | `str` | Full segment text. Stripped by all writers. |
| `start` | `float \| str` | Segment start in seconds (float from backend). Converted to formatted string by writers. |
| `end` | `float \| str` | Segment end. Same as `start`. |
| `score` | `float` | Segment-level confidence, computed as `round(math.exp(segment.avg_logprob), 2)`. Range approximately 0.0–1.0. |
| `words` | `list[WordData]` | All words in this segment with word-level timestamps. Always present; skipped if `segment.words is None` in the backend. |

> **Timestamp mutability warning:** The `float | str` union in both TypedDicts exists because `WriteJSON.transform_result` mutates the list in-place, replacing float timestamps with formatted strings before serialising. `WriteSRT` and `WriteVTT` do not mutate — they read floats via `cast(float, ...)` during iteration. Callers that need to reuse the result list after calling `writer.write()` should be aware that `WriteJSON` has side-effected it.

---

## 4. Abstract Base Classes

**File:** `src/wscribe/core.py`

Both base classes use `@dataclass(kw_only=True)`, enforcing keyword-only construction in all subclasses.

---

### `Backend`

The contract for all transcription inference backends.

```python
@dataclass(kw_only=True)
class Backend:
    name: str = "faster-whisper"
    model_size: str
```

| Member | Kind | Description |
|---|---|---|
| `name` | field (`str`) | Default `"faster-whisper"`. Identifies the backend. Not currently used programmatically beyond logging. |
| `model_size` | field (`str`) | **Required.** Validated in `__post_init__` against `supported_model_sizes()`. |
| `__post_init__` | concrete | Raises `ValueError` if `model_size` is not in `SUPPORTED_MODELS`. |
| `supported_backends()` | concrete | Returns `["faster-whisper"]`. No-op utility; intended for future multi-backend routing. |
| `supported_model_sizes()` | concrete | Returns `SUPPORTED_MODELS`. |
| `model_path()` | **abstract** | Must return the local filesystem path to the model directory. Raise `RuntimeError` if unavailable. |
| `load()` | **abstract** | Must initialise the model object and store it on `self`. |
| `transcribe(input: np.ndarray)` | **abstract** | Must accept a mono 16 kHz float32 numpy array and return `list[TranscribedData]`. |

---

### `Audio`

The contract for all audio sources.

```python
@dataclass(kw_only=True)
class Audio:
    source: str
    local_source_path: str = ""
    sampling_rate: int = 16000
```

| Member | Kind | Description |
|---|---|---|
| `source` | field (`str`) | **Required.** The source identifier — for local files this is the filesystem path. Future sources may use URLs. |
| `local_source_path` | field (`str`) | Initially empty. `fetch_audio()` is responsible for setting this to a local path that `convert_audio()` can read. |
| `sampling_rate` | field (`int`) | Default `16000`. Whisper's expected input sample rate. Passed to `decode_audio`. |
| `fetch_audio()` | **abstract** | Must validate/obtain the source and set `self.local_source_path`. |
| `determine_source_type(source)` | **abstract** (static) | Regex-match helper intended to identify the source type from a string. Not yet implemented in any subclass; intended for a future router. |
| `convert_audio()` | **concrete** | Calls `faster_whisper.audio.decode_audio(self.local_source_path, split_stereo=False, sampling_rate=self.sampling_rate)`. Returns a mono float32 numpy array at 16 kHz. This is the standard Whisper input format and the bridge between the source layer and backend layer. FFmpeg does the actual decoding transparently. |

---

## 5. Backends

**File:** `src/wscribe/backends/fasterwhisper.py`

### `FasterWhisperBackend`

The only currently implemented backend.

```python
DEFAULT_BEAM = 5

@dataclass(kw_only=True)
class FasterWhisperBackend(Backend):
    device: str = "cpu"        # "cpu" or "cuda"
    quantization: str = "int8" # "int8" (CPU) or "float16" (CUDA)
    model: WhisperModel | None = None
```

#### `model_path() -> str`

Constructs the expected path as:

```
$WSCRIBE_MODELS_DIR/faster-whisper-{model_size}
```

`os.environ["WSCRIBE_MODELS_DIR"]` is accessed directly — if the env var is unset, Python raises `KeyError`. If the path does not exist on disk, raises `RuntimeError`.

#### `load() -> None`

```python
self.model = WhisperModel(
    self.model_path(), device=self.device, compute_type=self.quantization
)
```

Stores the initialised `WhisperModel` on `self.model`. Must be called before `transcribe`.

#### `transcribe(input, language=None, silent=False, vad=False) -> list[TranscribedData]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | `np.ndarray` | required | Mono 16 kHz float32 array from `Audio.convert_audio()`. |
| `language` | `str \| None` | `None` | BCP-47 language code. `None` triggers faster-whisper's automatic language detection. |
| `silent` | `bool` | `False` | If `True`, disables the tqdm progress bar. |
| `vad` | `bool` | `False` | Enables faster-whisper's built-in VAD filter. Better results, slower. |

**Internals:**

1. Calls `self.model.transcribe(input, beam_size=5, word_timestamps=True, language=language, vad_filter=vad)` — returns a lazy generator of `Segment` objects plus an `info` object.
2. Opens a `tqdm` progress bar with `total=info.duration`, `unit="ps"` ("playback seconds"). This means the bar shows progress in terms of audio time consumed, not wall-clock time.
3. Iterates segments lazily. If `segment.words is None` the segment is skipped entirely.
4. For each segment, builds a `TranscribedData` dict:
   - `score` = `round(math.exp(segment.avg_logprob), 2)` — converts the log-probability (always ≤ 0) to a linear probability in [0, 1].
   - Words are extracted as `{"start": w.start, "end": w.end, "text": w.word, "score": round(w.probability, 2)}`.
5. Updates the progress bar by `segment.end - pbar.last_print_n` to advance in audio-time units.

**Device/quantisation convention** (enforced by the CLI, not the backend itself):

| Device | Quantisation |
|---|---|
| `cpu` | `int8` |
| `cuda` | `float16` |

**Performance numbers** (from README, single RTX 3050, 6.3 min audio):

| Device | Quantisation | Model | Transcription time | Speed |
|---|---|---|---|---|
| cuda | float16 | tiny | 0.1 m | 68× |
| cuda | float16 | small | 0.2 m | 29× |
| cuda | float16 | medium | 0.4 m | 14× |
| cuda | float16 | large-v2 | 0.8 m | 7× |
| cpu | int8 | tiny | 0.2 m | 25× |
| cpu | int8 | small | 1.3 m | 4× |
| cpu | int8 | medium | 3.6 m | ~1.7× |
| cpu | int8 | large-v2 | 3.6 m | ~0.9× |

### Planned Backends (Roadmap)

| Backend | Notes |
|---|---|
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | C++ implementation, likely lower memory footprint |
| [WhisperX](https://github.com/m-bain/whisperX) | Adds speaker diarization support |

### `MLXWhisperBackend`

**File:** `src/wscribe/backends/mlxwhisper.py`

Apple Silicon backend powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Requires the `[mlx]` optional extra: `uv tool install "wscribe[mlx]"`.

```python
@dataclass(kw_only=True)
class MLXWhisperBackend(Backend):
    name: str = "mlx-whisper"
```

#### Model identification

Unlike `FasterWhisperBackend`, models are not downloaded manually. The `model_size` field holds a Hugging Face repo ID from `MLX_SUPPORTED_MODELS` and the mlx-whisper library handles caching via the HF hub automatically.

```python
MLX_SUPPORTED_MODELS: list[str] = [
    "mlx-community/whisper-tiny-mlx",
    "mlx-community/whisper-tiny-mlx-q4",
    "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-small-mlx-q4",
    "mlx-community/whisper-medium-mlx",
    "mlx-community/whisper-medium-mlx-q4",
    "mlx-community/whisper-large-v2-mlx",
    "mlx-community/whisper-large-v2-mlx-4bit",
    "mlx-community/whisper-large-v3-mlx",
    "mlx-community/whisper-large-v3-mlx-4bit",
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-turbo-q4",
    "mlx-community/distil-whisper-large-v3",
]
```

#### Soft import pattern

`mlx_whisper` is imported at module level inside a `try/except ImportError` block and set to `None` on failure. This means the module can always be imported without raising — the `RuntimeError` is deferred to `load()` so that users on non-Apple-Silicon machines get a clear error message only when they actually try to use the backend.

#### `load() -> None`

Checks that `mlx_whisper is not None`; raises `RuntimeError` with install instructions if not. No model is explicitly loaded — mlx-whisper lazy-loads on the first `transcribe()` call.

#### `transcribe(input, language=None, silent=False, vad=False) -> list[TranscribedData]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | `np.ndarray` | required | Mono 16 kHz float32 array from `Audio.convert_audio()`. |
| `language` | `str \| None` | `None` | BCP-47 language code. `None` triggers auto-detection. |
| `silent` | `bool` | `False` | If `True`, passes `verbose=None` to suppress output. |
| `vad` | `bool` | `False` | **Ignored.** mlx-whisper has no VAD filter; a warning is logged. |

**Internals:**

1. Calls `mlx_whisper.transcribe(input, path_or_hf_repo=self.model_size, word_timestamps=True, language=language, verbose=...)`.
2. Iterates `raw["segments"]`; skips segments with no `words`.
3. Maps mlx-whisper's word dict keys: `w["word"]` → `"text"`, `w["probability"]` → `"score"`.
4. Segment-level score is the mean of its word scores.

#### Differences from `FasterWhisperBackend`

| Aspect | FasterWhisperBackend | MLXWhisperBackend |
|---|---|---|
| Hardware | CPU or CUDA | Apple Silicon only |
| Model storage | `$WSCRIBE_MODELS_DIR` (manual download) | HF hub cache (automatic) |
| Model identifier | Short name (`tiny`, `medium`, …) | HF repo ID (`mlx-community/...`) |
| Progress display | tqdm bar (audio-time units) | mlx-whisper's own output |
| VAD support | Yes (`vad_filter` param) | No (logs warning and ignores) |
| Install extra | *(always installed)* | `wscribe[mlx]` |

---

## 6. Sources

**File:** `src/wscribe/sources/local.py`

### `LocalAudio`

The only currently implemented source.

```python
@dataclass(kw_only=True)
class LocalAudio(Audio):
    def __post_init__(self):
        self.fetch_audio()
```

`__post_init__` immediately calls `fetch_audio()`, so the path is validated at construction time.

#### `fetch_audio()`

```python
def fetch_audio(self):
    if os.path.exists(self.source):
        self.local_source_path = self.source
    else:
        raise RuntimeError("specified local path doesn't exist")
```

Sets `self.local_source_path = self.source` if the path exists. Raises `RuntimeError` otherwise.

After construction, `audio = LocalAudio(source="/path/to/file").convert_audio()` produces the numpy array ready for backend consumption.

> **Note:** The CLI uses `click.Path(exists=True, dir_okay=False)` for the SOURCE argument, so path validation is actually double-checked — once by Click before `LocalAudio` is ever instantiated, and once inside `fetch_audio`.

### Planned Sources (Roadmap)

| Source | Notes |
|---|---|
| YouTube | URL-based source; would download to a temp local path before `convert_audio` |
| Google Drive | Similar pattern — fetch to local, then convert |

The `Audio.determine_source_type(source: str)` static method is stubbed (raises `NotImplementedError`) and is intended as the future router that would select the appropriate `Audio` subclass from a raw source string.

---

## 7. Writers

**File:** `src/wscribe/writers.py`

Attribution comment: `# Based on code from https://github.com/openai/whisper`

### `format_timestamp(seconds: float, decimal_marker: str = ".") -> str`

Converts a float number of seconds to `HH:MM:SS{decimal_marker}mmm`.

```python
format_timestamp(6.92)          # "00:00:06.920"
format_timestamp(6.92, ",")     # "00:00:06,920"  (SRT style)
```

The decimal marker is the only difference between SRT and VTT/JSON timestamp formats.

---

### `ResultWriter` (abstract base)

```python
@dataclass(kw_only=True)
class ResultWriter:
    result: list[TranscribedData]
    destination: os.PathLike
```

| Member | Kind | Description |
|---|---|---|
| `result` | field | The transcription data to write. |
| `destination` | field | Output file path. |
| `write()` | concrete | Opens `destination` with UTF-8 encoding, then calls `_write_result(self.result, f)`. This is the **Template Method**. |
| `_write_result(result, file)` | **abstract** | Must write the formatted output to the file handle. |

---

### `SubtitlesWriter` (intermediate base for SRT and VTT)

```python
@dataclass(kw_only=True)
class SubtitlesWriter(ResultWriter):
    decimal_marker: str
```

Adds `decimal_marker` (set by concrete subclasses) and a shared generator:

#### `iterate_result(result) -> Generator`

Yields `(segment_start: str, segment_end: str, segment_text: str)` tuples for each segment. Strips whitespace from `segment_text`. Calls `self.format_timestamp` (which delegates to the module-level `format_timestamp` with the instance's `decimal_marker`).

---

### `WriteSRT`

```python
@dataclass(kw_only=True)
class WriteSRT(SubtitlesWriter):
    decimal_marker: str = ","
```

SRT format: numbered blocks, comma as decimal separator.

```
1
00:00:00,000 --> 00:00:06,920
In your work, you quote Nietzsche quite a bit, his line,

2
00:00:07,120 --> 00:00:10,540
he who has a why to live for can bear almost any how.
```

`_write_result` iterates using `enumerate(..., start=1)` and prints `{i}\n{start} --> {end}\n{text}\n`.

---

### `WriteVTT`

```python
@dataclass(kw_only=True)
class WriteVTT(SubtitlesWriter):
    decimal_marker: str = "."
```

WebVTT format: `WEBVTT\n` header, period as decimal separator, no block numbers.

```
WEBVTT

00:00:00.000 --> 00:00:06.920
In your work, you quote Nietzsche quite a bit, his line,

00:00:07.120 --> 00:00:10.540
he who has a why to live for can bear almost any how.
```

`_write_result` prints the header then iterates using `iterate_result`.

---

### `WriteJSON`

```python
@dataclass(kw_only=True)
class WriteJSON(ResultWriter):
    pass  # uses default decimal_marker="." via format_timestamp default
```

JSON format — the richest export, preserving word-level data.

`_write_result` calls `self.transform_result(result)` first, then `json.dump(result, file)`.

#### `transform_result(result)` — **mutates in place**

Iterates over every segment and every word in `result`, replacing:
- `s["start"]` / `s["end"]` — float → `"HH:MM:SS.mmm"` string
- `s["text"]` — stripped
- `w["start"]` / `w["end"]` — float → `"HH:MM:SS.mmm"` string
- `w["text"]` — stripped

> **Side-effect warning:** After `WriteJSON.write()` returns, the original `result` list has been mutated — all timestamps are now strings, not floats. Do not reuse the list for further processing that expects floats.

#### Output structure

```json
[
  {
    "text": "In your work, you quote Nietzsche quite a bit, his line,",
    "start": "00:00:00.000",
    "end": "00:00:06.920",
    "score": 0.85,
    "words": [
      { "start": "00:00:00.000", "end": "00:00:00.980", "text": "In", "score": 0.53 },
      { "start": "00:00:01.000", "end": "00:00:01.320", "text": "your", "score": 0.99 }
    ]
  }
]
```

---

### `WRITERS` registry

```python
WRITERS: Mapping[str, typing.Type[ResultWriter]] = {
    "json": WriteJSON,
    "srt": WriteSRT,
    "vtt": WriteVTT,
}
```

The CLI and the `--format` option both derive their valid choices directly from `WRITERS.keys()`. To add a new format, add a new key here.

---

### Format comparison

| Feature | JSON | SRT | VTT |
|---|---|---|---|
| Word-level timestamps | ✅ | ❌ | ❌ |
| Per-word confidence scores | ✅ | ❌ | ❌ |
| Segment confidence scores | ✅ | ❌ | ❌ |
| Decimal separator | `.` | `,` | `.` |
| Block numbers | N/A | ✅ | ❌ |
| File header | None | None | `WEBVTT` |
| Mutates input list | ✅ yes | ❌ no | ❌ no |
| Suitable for wscribe-editor | ✅ | ❌ | ❌ |

---

## 8. CLI

**File:** `src/wscribe/cli/main.py`  
**Entry point:** `wscribe = "wscribe.cli.main:cli"` (defined in `pyproject.toml`)

### Command group

```
wscribe [COMMAND]
```

| Command | Description |
|---|---|
| `transcribe` | Main transcription workflow |
| `info` | Print `WSCRIBE_MODELS_DIR` value |

---

### `wscribe transcribe SOURCE DESTINATION [OPTIONS]`

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `SOURCE` | — | `Path` (must exist, not a dir, resolved) | required | Input audio/video file |
| `DESTINATION` | — | `Path` (must not exist, resolved) | required | Output file path |
| `--format` | `-f` | Choice: `json`, `srt`, `vtt` | `json` | Output format |
| `--model` | `-m` | Choice: `tiny`, `small`, `medium`, `large-v2` | `medium` | Whisper model size |
| `--gpu` | `-g` | flag | `False` | Use CUDA; sets `device=cuda`, `quantization=float16` |
| `--language` | `-l` | `str` | `None` | BCP-47 language code (e.g. `en`, `fr`); skips autodetection |
| `--debug` | `-d` | flag | `False` | Forces `logging.DEBUG` level |
| `--stats` | `-s` | flag | `False` | Print timing/speed summary line after completion |
| `--quiet` | `-q` | flag | `False` | Suppress tqdm progress bar |
| `--vad` | `-v` | flag | `False` | Enable VAD filter in faster-whisper |

#### Execution flow

1. If `--debug`: `logging.basicConfig(level=logging.DEBUG, force=True)`
2. Bind source/destination/format/model/gpu to the structlog context.
3. Determine `(device, quantization)`: `("cuda", "float16")` if `--gpu`, else `("cpu", "int8")`.
4. `FasterWhisperBackend(model_size=model, device=device, quantization=quantization).load()`
5. Time `LocalAudio(source=source).convert_audio()` → numpy array (`audio_start_time` … `audio_end_time`).
6. Time `m.transcribe(input=audio, language=language, silent=quiet, vad=vad)` → `result` (`ts_start_time` … `ts_end_time`).
7. `WRITERS[format](result=result, destination=destination).write()`
8. If `--stats`: compute and print the stats line (see below).

#### Stats output format

When `--stats` is passed, a single pipe-delimited line is printed to stdout:

```
cpu|int8|medium|1.0s|6.3m|3.6m|~1x
```

| Field | Value |
|---|---|
| 1 | `device` (`cpu` or `cuda`) |
| 2 | `quantization` (`int8` or `float16`) |
| 3 | `model` size |
| 4 | Audio conversion time (seconds, 1 d.p., e.g. `1.0s`) |
| 5 | Original audio playback duration (minutes, 1 d.p., e.g. `6.3m`) — computed as `audio.shape[0] / LocalAudio.sampling_rate / 60` |
| 6 | Transcription time (minutes, 1 d.p., e.g. `3.6m`) |
| 7 | Speed multiplier (integer, e.g. `~1x`) — `int(original_audio_time / transcription_time)` |

> **Note:** `LocalAudio.sampling_rate` is accessed as a class attribute (default `16000`). This is safe because `sampling_rate` has a default value in the dataclass definition and is never overridden in practice.

---

### `wscribe info`

Prints the value of `WSCRIBE_MODELS_DIR`:

```
WSCRIBE_MODELS_DIR: /home/user/.local/share/whisper-models
```

Raises `KeyError` if the env var is unset.

---

### Usage examples

```bash
# CPU, medium model, JSON output (all defaults)
wscribe transcribe audio.mp3 transcription.json

# GPU, default format (JSON)
wscribe transcribe video.mp4 transcription.json --gpu

# GPU, SRT format
wscribe transcribe video.mp4 transcription.srt -g -f srt

# GPU, WebVTT, tiny model
wscribe transcribe video.mp4 transcription.vtt -g -f vtt -m tiny

# CPU, specific language, with stats
wscribe transcribe audio.mp3 out.json -l en -s

# GPU, VAD filter, quiet (no progress bar), print stats — benchmark pattern
wscribe transcribe test.mp3 test.json -m medium -g -q -s

# Show model directory
wscribe info
```

---

## 9. Logging

**File:** `src/wscribe/__init__.py`

Logging is configured at module import time (the two `setup_*` calls at the bottom of the file execute unconditionally).

### `setup_stdlogger()`

Configures the Python stdlib `logging` root logger:
- Format: `"%(message)s"` (structlog renders its own format)
- Stream: `sys.stdout` (following [12-factor app log guidance](https://12factor.net/logs))
- Level: `None` (no level filter at the stdlib layer; structlog controls filtering)

### `get_structlog_processors() -> Iterable[Processor]`

Builds the processor chain:

| Processor | Effect |
|---|---|
| `structlog.stdlib.add_logger_name` | Adds `logger` field |
| `structlog.stdlib.add_log_level` | Adds `level` field |
| `structlog.stdlib.PositionalArgumentsFormatter` | Formats `%`-style positional args |
| `TimeStamper(fmt="%Y-%m-%d %H:%M.%S")` | Adds human-readable local timestamp |
| `TimeStamper(fmt="iso")` | Adds ISO 8601 timestamp |
| `StackInfoRenderer` | Renders stack info if present |
| `format_exc_info` | Renders exception tracebacks |
| `UnicodeDecoder` | Decodes byte strings |
| `CallsiteParameterAdder({FILENAME, FUNC_NAME, LINENO})` | Adds source location metadata |
| *(TTY)* `ConsoleRenderer` | Human-readable coloured output |
| *(non-TTY)* `dict_tracebacks` + `JSONRenderer` | Structured JSON lines (for log aggregation pipelines) |

The renderer selection is based on `sys.stderr.isatty()`: if stderr is a terminal, use `ConsoleRenderer`; otherwise use JSON. This means piping `wscribe` output or running it in CI automatically produces machine-parseable logs.

### `setup_structlog()`

Calls `structlog.configure` with:
- `logger_factory=structlog.stdlib.LoggerFactory()` — bridges structlog to stdlib
- `cache_logger_on_first_use=True` — performance optimisation; locks processor chain after first use
- `wrapper_class=structlog.BoundLogger`
- `processors=get_structlog_processors()`

### Usage in modules

Each module gets its own named logger:

```python
LOGGER = structlog.get_logger()                    # core, backends, sources
LOGGER = structlog.get_logger(ui="cli")            # cli/main.py — binds ui="cli" permanently
```

Context is bound per-call in the CLI:

```python
log = LOGGER.bind(source=source, destination=destination, format=format, model=model, gpu=gpu)
log.debug("model loaded with {device}-{quantization}")
```

---

## 10. Configuration & Packaging

**File:** `pyproject.toml`

### Package metadata

| Field | Value |
|---|---|
| Name | `wscribe` |
| Version | `0.1.5` |
| Python | `>=3.10` |
| Build backend | `hatchling` |
| Package root | `src/wscribe` |
| PyPI readme | `docs/README.md` |

### Runtime dependencies

| Package | Version constraint | Role |
|---|---|---|
| `structlog` | `>=23.1.0` | Structured logging |
| `faster-whisper` | `>=0.9.0` | Inference engine (also provides `decode_audio`) |
| `click` | `>=8.1.6` | CLI framework |

> `numpy`, `tqdm`, and `ctranslate2` are transitive dependencies pulled in by `faster-whisper`.

### Optional dependencies

| Extra | Package | Purpose |
|---|---|---|
| `mlx` | `mlx-whisper>=0.4.3` | Apple Silicon backend; install with `uv tool install "wscribe[mlx]"` |

### Dev dependencies (group `dev`)

| Package | Purpose |
|---|---|
| `pudb` | TUI debugger |
| `ipython` | Interactive shell |
| `isort` | Import sorting (used in `make lint`) |
| `ruff` | Linter (currently commented out in Makefile) |
| `black` | Code formatter (used in `make lint`) |
| `snoop` | Print-based tracing/debugging |

### Test dependencies (group `test`)

| Package | Purpose |
|---|---|
| `pytest` | `>=7.3.1` — test runner |
| `mypy` | `>=1.3.0` — static type checker (used in `make typecheck`) |

### pytest configuration

```toml
[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"
```

Live log output is always enabled at INFO level during test runs.

### uv

The project uses [uv](https://docs.astral.sh/uv/) as its package manager. uv is a fast, single-binary Rust-based tool that replaces `pip`, `pip-tools`, `venv`, and `poetry` for this project.

**Install uv:**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Key workflows:**

| Task | Command |
|---|---|
| Install all deps (core + dev + test) | `uv sync --all-groups` |
| Install with mlx extra (Apple Silicon) | `uv sync --all-groups --extra mlx` |
| Run a script in the project venv | `uv run pytest` |
| Install as a standalone tool | `uv tool install wscribe` |
| Install with mlx extra as a tool | `uv tool install "wscribe[mlx]"` |
| Build sdist + wheel | `uv build` |
| Publish to PyPI | `uv publish` |
| Publish to Test PyPI | `uv publish --index testpypi` |
| Regenerate lockfile | `uv lock` |

**Lockfile:** `uv.lock` is committed to the repository. It is a cross-platform lockfile that pins all transitive dependencies. Unlike `poetry.lock`, it is a TOML file and human-readable.

**Test PyPI index** is configured in `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
publish-url = "https://test.pypi.org/legacy/"
explicit = true
```

---

## 11. Tooling

### Makefile targets

| Target | Command | Description |
|---|---|---|
| `deps-sync` | `uv sync --all-groups` | Sync all dependencies from lockfile |
| `deps-sync-mlx` | `uv sync --all-groups --extra mlx` | Sync deps including Apple Silicon mlx extra |
| `deps-show` | `uv pip list` | List all installed packages |
| `package-publish` | `uv publish` | Publish to PyPI |
| `package-publish-test` | `uv publish --index testpypi` | Publish to Test PyPI |
| `package-build` | `uv build` | Build sdist + wheel |
| `package-version-bump-patch` | `uv version --bump patch` | Increment patch version |
| `package-version-bump-prerelease` | `uv version --bump patch --pre alpha` | Increment prerelease version |
| `spin` | `typecheck lint test-quiet` | Full CI check suite (type check + lint + quiet tests) |
| `test` | `uv run pytest` | Run all tests with verbose output |
| `typecheck` | `uv run mypy .` | Run static type checker |
| `lint` | `uv run isort --check-only` + `uv run black --check` | Check import ordering and code formatting (ruff is commented out) |
| `test-quiet` | `uv run pytest -q` | Run tests quietly |
| `test-dry-run` | `uv run pytest --collect-only` | List all test names without running them |
| `help` | *(auto-generated)* | Lists all `.PHONY` targets that have a `# comment` annotation |

### Makefile.common

Shared utilities included at the bottom of `Makefile`.

| Target | Description |
|---|---|
| `update-makefile-common` | Downloads the latest `Makefile.common` from `https://raw.githubusercontent.com/geekodour/t/main/Makefile.common` |
| `show-template-placeholder` | Uses `rg` and `fd` to find unreplaced `bake` template placeholders in the workspace |
| `help` | Sets `DEFAULT_GOAL`; auto-generates help by parsing `# comment` annotations on `.PHONY` targets |

### `scripts/fw_dw_hf_wo_lfs.sh` — Model Downloader

Downloads a faster-whisper model from HuggingFace without requiring `git lfs` to be installed.

**Prerequisites:** `WSCRIBE_MODELS_DIR` env var must be set.

**Usage:**

```bash
./scripts/fw_dw_hf_wo_lfs.sh tiny    # also: small, medium, large-v2
```

**Flow:**

1. Validates `WSCRIBE_MODELS_DIR` is set; exits with error if not.
2. Validates the argument matches `(tiny|small|medium|large-v2)`.
3. Constructs `hf_url = "https://huggingface.co/guillaumekln/faster-whisper-{size}"`.
4. `git clone "$hf_url" "$WSCRIBE_MODELS_DIR/faster-whisper-{size}"` — clones the repo (without LFS binary).
5. Attempts `git lfs pull`; if that fails, falls back to `curl -L "$hf_url/resolve/main/model.bin"` to download the model binary directly.
6. The result is a directory at `$WSCRIBE_MODELS_DIR/faster-whisper-{size}` containing the model files.

### `scripts/speed_check.sh` — Benchmark

Runs `wscribe transcribe` for all 8 combinations (4 models × GPU then CPU) against a `./test.mp3` file, printing one stats line per run.

```bash
# GPU runs first (tiny → large-v2)
wscribe transcribe ./test.mp3 test.json -m tiny     -g -q -s
wscribe transcribe ./test.mp3 test.json -m small    -g -q -s
wscribe transcribe ./test.mp3 test.json -m medium   -g -q -s
wscribe transcribe ./test.mp3 test.json -m large-v2 -g -q -s

# CPU runs
wscribe transcribe ./test.mp3 test.json -m tiny     -q -s
wscribe transcribe ./test.mp3 test.json -m small    -q -s
wscribe transcribe ./test.mp3 test.json -m medium   -q -s
wscribe transcribe ./test.mp3 test.json -m large-v2 -q -s
```

The `-q` (quiet/no progress bar) and `-s` (stats) flags make the output machine-readable for comparison. The output file `test.json` is overwritten on each run.

---

## 12. Tests

**File:** `tests/test_wscribe.py`

### Fixture: `faster_whisper_tools`

```python
@pytest.fixture
def faster_whisper_tools():
    model = FasterWhisperBackend(model_size="tiny", device="cpu", quantization="int8")
    sample_audio_path = os.path.join(
        os.environ["PROJECT_ROOT"], "examples", "assets", "jfk.wav"
    )
    return model, sample_audio_path
```

**Required environment variables:**

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT` | Absolute path to the repo root; used to locate `examples/assets/jfk.wav` |
| `WSCRIBE_MODELS_DIR` | Used by `FasterWhisperBackend.model_path()` to find `faster-whisper-tiny` |

### Test: `TestFastWhisper.test_json`

```python
def test_json(self, faster_whisper_tools):
    model, sample_path = faster_whisper_tools
    audio = LocalAudio(source=sample_path).convert_audio()
    model.load()
    data = model.transcribe(input=audio)
    assert set(data[0].keys()) == {"text", "start", "end", "score", "words"}
    assert set(data[0]["words"][0].keys()) == {"text", "start", "end", "score"}
```

This is an end-to-end integration test of the source → backend pipeline:

1. `LocalAudio(source=sample_path).convert_audio()` — exercises path validation and FFmpeg decoding.
2. `model.load()` — exercises `model_path()` and `WhisperModel` initialisation.
3. `model.transcribe(input=audio)` — exercises the full inference pass.
4. Asserts that the `TranscribedData` and `WordData` TypedDicts have exactly the expected keys.

### Coverage gaps

The following are **not tested**:

| Area | Missing coverage |
|---|---|
| Writers | No tests for `WriteJSON`, `WriteSRT`, `WriteVTT` output formatting or the `WRITERS` registry |
| `WriteJSON.transform_result` | The in-place timestamp mutation is untested |
| `format_timestamp` | Edge cases (zero, rounding, values > 1 hour) |
| CLI commands | No tests for `transcribe` or `info` via Click's test runner |
| `LocalAudio` path validation | The `RuntimeError` on missing path is untested |
| `Backend.__post_init__` | The `ValueError` on invalid model size is untested |
| GPU path | No GPU tests exist |
| `--vad`, `--language` flags | Not exercised in tests |
| `--stats` output format | Not tested |

---

## 13. Design Patterns & Extension Points

### Patterns in use

| Pattern | Location | How it's applied |
|---|---|---|
| **Abstract Base Class** | `Backend`, `Audio` in `core.py` | `@dataclass(kw_only=True)` classes with methods that `raise NotImplementedError`. No use of `abc.ABC` — conventions rather than enforcement. |
| **Template Method** | `ResultWriter.write()` / `_write_result()` | `write()` handles file opening; subclasses implement only `_write_result`. |
| **Registry / Plugin Map** | `WRITERS` dict in `writers.py` | Maps string format keys to writer classes. CLI reads keys for `click.Choice`; instantiation is `WRITERS[format](...)`. |
| **Strategy** | Sources, Backends, Writers as layers | Each layer is independently swappable. The CLI wires them together; none of the three layers knows about the others. |
| **Structured Logging** | `__init__.py`, all modules | structlog with context binding; automatically switches between human-readable (TTY) and JSON-lines (pipeline/CI) rendering. |
| **TypedDict contracts** | `WordData`, `TranscribedData` | Lightweight typed data contracts between layers without ORM or Pydantic overhead. |
| **Dataclass composition** | All classes | `@dataclass(kw_only=True)` throughout for clean field declarations and `__post_init__` hooks. |

---

### Adding a new Backend

1. Create `src/wscribe/backends/mybackend.py`.
2. Subclass `Backend`:
   ```python
   from ..core import Backend, TranscribedData
   import numpy as np

   @dataclass(kw_only=True)
   class MyBackend(Backend):
       name: str = "my-backend"
       # add your fields here

       def supported_model_sizes(self) -> list[str]:
           return ["model-a", "model-b"]  # or a module-level constant

       def model_path(self) -> str:
           # return path/ID for the model; raise RuntimeError if unavailable

       def load(self) -> None:
           # initialise self.model (or soft-import and check here)

       def transcribe(self, input: np.ndarray, **kwargs) -> list[TranscribedData]:
           # run inference; return list[TranscribedData]
   ```
3. If the backend has optional system dependencies, use the soft-import pattern:
   ```python
   try:
       import my_lib
   except ImportError:
       my_lib = None
   ```
   Then raise a helpful `RuntimeError` in `load()` when `my_lib is None`.
4. Expose any new model identifiers as a module-level constant (e.g. `MY_SUPPORTED_MODELS`) so the CLI can import them for validation.
5. Wire up in `src/wscribe/cli/main.py`:
   - Add the new backend name to the `click.Choice` list on `--backend`.
   - Import the backend class and its model list.
   - Add an `elif backend == "my-backend":` branch in the `transcribe` command to validate the model and instantiate the class.
   - Update the `stats` block if the backend uses different device/quantisation concepts.
   - Update `info()` to list the new backend's supported models.
6. If the backend requires an optional PyPI dependency, add it under `[project.optional-dependencies]` in `pyproject.toml` and re-run `uv lock`.

---

### Adding a new Source

1. Create `src/wscribe/sources/mysource.py`.
2. Subclass `Audio`:
   ```python
   from ..core import Audio

   @dataclass(kw_only=True)
   class MySource(Audio):
       def __post_init__(self):
           self.fetch_audio()

       def fetch_audio(self):
           # download/copy the resource to a local temp path
           self.local_source_path = "/tmp/downloaded_audio.wav"
   ```
3. `convert_audio()` is inherited and works unchanged — it only needs `self.local_source_path` to be set.
4. Import and instantiate in `cli/main.py` (either hardcode or add a `--source-type` CLI option).

---

### Adding a new Writer

1. Add a class to `src/wscribe/writers.py` that subclasses `ResultWriter` (or `SubtitlesWriter` for subtitle formats).
2. Implement `_write_result(self, result, file)`.
3. Register it in `WRITERS`:
   ```python
   WRITERS: Mapping[str, typing.Type[ResultWriter]] = {
       "json": WriteJSON,
       "srt": WriteSRT,
       "vtt": WriteVTT,
       "myformat": WriteMyFormat,   # add here
   }
   ```
4. The CLI's `--format` `click.Choice` and all routing are automatically derived from `WRITERS.keys()` — no other changes required.
