import json
import logging
import os
import time

import click
import structlog

from wscribe.backends.fasterwhisper import FasterWhisperBackend
from wscribe.backends.mlxwhisper import MLX_SUPPORTED_MODELS, MLXWhisperBackend
from wscribe.sources.local import LocalAudio

from ..core import SUPPORTED_MODELS
from ..writers import WRITERS

LOGGER = structlog.get_logger(ui="cli")


@click.group()
def cli():
    """CLI for audio transcription (faster-whisper or mlx-whisper backends)"""
    pass


@cli.command()
@click.argument(
    "source",
    nargs=1,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.argument(
    "destination",
    nargs=1,
    type=click.Path(exists=False, resolve_path=True),
)
@click.option(
    "-f",
    "--format",
    help="destication file format, currently only json is supported",
    type=click.Choice(list(WRITERS.keys()), case_sensitive=True),
    default="json",
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    help="model size or HF repo ID. faster-whisper: tiny/small/medium/large-v2. mlx-whisper: mlx-community/... repo ID.",
    type=click.STRING,
    default="medium",
    show_default=True,
)
@click.option(
    "-b",
    "--backend",
    help="transcription backend to use",
    type=click.Choice(["faster-whisper", "mlx-whisper"], case_sensitive=True),
    default="faster-whisper",
    show_default=True,
)
@click.option(
    "-g", "--gpu", help="enable gpu, disabled by default", default=False, is_flag=True
)
@click.option("-l", "--language", help="language code eg. en/fr (skips autodetection)")
@click.option("-d", "--debug", help="show debug logs", default=False, is_flag=True)
@click.option("-s", "--stats", help="print stats", default=False, is_flag=True)
@click.option("-q", "--quiet", help="no progress bar", default=False, is_flag=True)
@click.option(
    "-v",
    "--vad",
    help="use vad filter(better results, slower)",
    default=False,
    is_flag=True,
)
def transcribe(
    source, destination, format, model, backend, gpu, language, debug, stats, quiet, vad
):
    """
    Transcribes SOURCE to DESTINATION. Where SOURCE can be local path to an audio/video file and
    DESTINATION needs to be a local path to a non-existing file.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG, force=True)
    log = LOGGER.bind(
        source=source, destination=destination, format=format, model=model, gpu=gpu
    )

    if backend == "mlx-whisper":
        if model not in MLX_SUPPORTED_MODELS:
            raise click.BadParameter(
                f"For --backend mlx-whisper, --model must be one of:\n  "
                + "\n  ".join(MLX_SUPPORTED_MODELS),
                param_hint="'--model'",
            )
        m = MLXWhisperBackend(model_size=model)
        m.load()
        log.debug("mlx-whisper model ready", model=model)
    else:
        if model not in SUPPORTED_MODELS:
            raise click.BadParameter(
                f"For --backend faster-whisper, --model must be one of: {SUPPORTED_MODELS}",
                param_hint="'--model'",
            )
        device, quantization = ("cuda", "float16") if gpu else ("cpu", "int8")
        m = FasterWhisperBackend(model_size=model, device=device, quantization=quantization)
        m.load()
        log.debug(f"model loaded with {device}-{quantization}")

    audio_start_time = time.perf_counter()
    audio = LocalAudio(source=source).convert_audio()
    audio_end_time = time.perf_counter()

    ts_start_time = time.perf_counter()
    result = m.transcribe(input=audio, language=language, silent=quiet, vad=vad)
    ts_end_time = time.perf_counter()

    writer = WRITERS[format](result=result, destination=destination)
    writer.write()

    if stats:
        original_audio_time = audio.shape[0] / LocalAudio.sampling_rate
        transcription_time = ts_end_time - ts_start_time
        audio_conversion_time = audio_end_time - audio_start_time
        if backend == "mlx-whisper":
            backend_tag = "mlx|apple-silicon"
        else:
            backend_tag = f"{device}|{quantization}"
        click.echo(
            " | ".join(
                [
                    backend_tag,
                    model,
                    str(round(audio_conversion_time, 1)) + "s",
                    str(round(original_audio_time / 60, 1)) + "m",
                    str(round(transcription_time / 60, 1)) + "m",
                    str(int(original_audio_time / transcription_time)) + "x",
                ]
            )
        )


@cli.command()
def info():
    """Information about related files and directories"""
    click.echo(f"WSCRIBE_MODELS_DIR: {os.environ.get('WSCRIBE_MODELS_DIR', '(not set)')}")
    click.echo("Available MLX models (--backend mlx-whisper):")
    for repo in MLX_SUPPORTED_MODELS:
        click.echo(f"  {repo}")


if __name__ == "__main__":
    cli()
