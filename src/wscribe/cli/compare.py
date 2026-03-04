import json
from pathlib import Path
from typing import Any

import click
import structlog

from ..compare import compare_transcriptions
from ..writers import WriteJSON

LOGGER = structlog.get_logger(ui="cli")


@click.command()
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(exists=False, resolve_path=True),
    help="Path to write the comparison JSON",
)
@click.option(
    "--time-tolerance",
    default=0.2,
    show_default=True,
    type=float,
    help="Seconds within which two words are considered to refer to the same audio moment",
)
def compare(files: tuple[str, ...], output: str, time_tolerance: float) -> None:
    """Compare two or more wscribe JSON transcriptions of the same audio.

    Produces a single JSON file where each word score reflects inter-transcription
    agreement. Low-scoring words are likely transcription errors.
    """
    if len(files) < 2:
        raise click.UsageError("compare requires at least 2 input files")

    transcriptions: list[Any] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise click.ClickException(f"{path}: expected a JSON array")
            transcriptions.append(data)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"{path}: invalid JSON — {e}") from e

    LOGGER.info("comparing transcriptions", n=len(transcriptions), tolerance=time_tolerance)
    result = compare_transcriptions(transcriptions, tolerance=time_tolerance)

    writer = WriteJSON(result=result, destination=Path(output))
    writer.write()
    LOGGER.info("written", output=output)
