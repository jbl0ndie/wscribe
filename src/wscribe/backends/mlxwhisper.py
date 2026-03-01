"""
MLX Whisper backend for Apple Silicon (M1/M2/M3).

Install the optional dependency before using this backend:

    uv tool install "wscribe[mlx]"
    # or, inside a project:
    uv sync --extra mlx
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

# Soft-import: mlx_whisper is only available when the [mlx] extra is installed.
# Importing it at module level would break non-Apple-Silicon machines entirely.
try:
    import mlx_whisper  # type: ignore
except ImportError:
    mlx_whisper = None  # type: ignore

from ..core import Backend, TranscribedData

LOGGER = structlog.get_logger()

# Full list of publicly available mlx-community Whisper repos on Hugging Face.
# Pass any of these strings as the --model argument when using --backend mlx-whisper.
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


@dataclass(kw_only=True)
class MLXWhisperBackend(Backend):
    """
    Transcription backend powered by mlx-whisper on Apple Silicon.

    Unlike FasterWhisperBackend, models are downloaded automatically from
    Hugging Face on first use — there is no separate model download step and
    no WSCRIBE_MODELS_DIR env var required.

    Attributes
    ----------
    model_size:
        A Hugging Face repo ID from MLX_SUPPORTED_MODELS, e.g.
        "mlx-community/whisper-large-v3-turbo".
    """

    name: str = "mlx-whisper"

    def supported_model_sizes(self) -> list[str]:
        return MLX_SUPPORTED_MODELS

    def model_path(self) -> str:
        # For mlx-whisper the model identifier IS the HF repo ID; the library
        # handles downloading/caching automatically via the HF hub.
        return self.model_size

    def load(self) -> None:
        if mlx_whisper is None:
            raise RuntimeError(
                "mlx-whisper is not installed. "
                "Install the optional extra with:\n\n"
                "    uv tool install \"wscribe[mlx]\"\n\n"
                "Note: mlx-whisper requires Apple Silicon (M1/M2/M3)."
            )
        # No explicit load step needed — mlx_whisper.transcribe() loads the
        # model lazily on first call.

    def transcribe(
        self,
        input: np.ndarray,
        language: Optional[str] = None,
        silent: bool = False,
        vad: bool = False,
    ) -> list[TranscribedData]:
        """
        Return word-level transcription data using mlx-whisper.

        Parameters
        ----------
        input:
            Audio samples as a float32 numpy array (16 kHz mono), as produced
            by Audio.convert_audio().
        language:
            BCP-47 language code (e.g. "en", "fr").  None triggers
            auto-detection.
        silent:
            When True, suppress the mlx-whisper progress output.
        vad:
            Ignored by this backend (mlx-whisper does not expose a VAD filter
            directly).  A warning is logged when True.
        """
        if vad:
            LOGGER.warning(
                "mlx-whisper backend does not support the --vad flag; ignoring"
            )

        raw = mlx_whisper.transcribe(
            input,
            path_or_hf_repo=self.model_size,
            word_timestamps=True,
            language=language,
            verbose=None if silent else False,
        )

        result: list[TranscribedData] = []
        for segment in raw.get("segments", []):
            words = segment.get("words", [])
            if not words:
                continue
            word_data = [
                {
                    "text": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "score": round(float(w.get("probability", 0.0)), 2),
                }
                for w in words
            ]
            result.append(
                {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "score": round(
                        sum(w["score"] for w in word_data) / len(word_data), 2
                    ),
                    "words": word_data,
                }
            )

        return result
