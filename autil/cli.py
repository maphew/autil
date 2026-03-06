"""CLI interface for autil."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .analyzer import analyze_audio, format_summary, save_results
from .audio_loader import get_audio_info


app = typer.Typer(
    name="autil",
    help="Audio Utilities - CLI-first audio analysis tools",
    add_completion=False,
    pretty_exceptions_short=True,
)


def version_callback(
    version: bool = typer.Option(None, "--version", is_eager=True, help="Show version"),
) -> None:
    """Show version information."""
    if version:
        console.print(f"autil version {__version__}")
        raise typer.Exit()


app.callback(invoke_without_command=True)(version_callback)
console = Console()


@app.command()
def fingerprint(
    version: Optional[bool] = typer.Option(
        None, "--version", is_eager=True, help="Show version"
    ),
    input: str = typer.Argument(..., help="Input audio file"),
    output_dir: Optional[str] = typer.Option(
        None, "-o", "--output-dir", help="Output directory (default: input directory)"
    ),
    silence_threshold: float = typer.Option(
        -40.0, "--silence-threshold", help="Silence threshold in dB"
    ),
    sensitivity: float = typer.Option(
        0.5, "--sensitivity", help="Detection sensitivity 0.0-1.0"
    ),
    json_only: bool = typer.Option(
        False, "--json-only", help="Skip visualization image"
    ),
    json: Optional[str] = typer.Option(
        None, "-j", "--json", help="Custom JSON output path"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed output"),
) -> None:
    """Analyze audio file and produce fingerprint."""
    if version:
        console.print(f"autil version {__version__}")
        raise typer.Exit()

    input_path = Path(input)

    # Validate input
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input}")
        raise typer.Exit(1)

    if not input_path.is_file():
        console.print(f"[red]Error:[/red] Input is not a file: {input}")
        raise typer.Exit(1)

    # Validate sensitivity
    if not 0.0 <= sensitivity <= 1.0:
        console.print(f"[red]Error:[/red] Sensitivity must be between 0.0 and 1.0")
        raise typer.Exit(1)

    # Resolve output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path.parent

    # Default JSON path
    if json:
        json_path = Path(json)
    else:
        json_path = output_path / f"{input_path.stem}_fingerprint.json"

    # Run analysis
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing audio...", total=None)

            results = analyze_audio(
                str(input_path),
                silence_threshold_db=silence_threshold,
                sensitivity=sensitivity,
                create_viz=not json_only,
                output_dir=output_dir,
            )

            progress.update(task, completed=True)

        # Save results
        save_results(results, str(json_path))

        # Print summary
        console.print(format_summary(results))

        # Show file paths
        console.print(f"\n[green]Saved:[/green] {json_path}")
        if results.get("visualization"):
            console.print(f"[green]Saved:[/green] {results['visualization']}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def info(
    version: Optional[bool] = typer.Option(
        None, "--version", is_eager=True, help="Show version"
    ),
    input: str = typer.Argument(..., help="Input audio file"),
) -> None:
    """Show audio file information."""
    if version:
        console.print(f"autil version {__version__}")
        raise typer.Exit()

    input_path = Path(input)

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input}")
        raise typer.Exit(1)

    try:
        info = get_audio_info(str(input_path))

        console.print(f"Duration: {info['duration']:.2f}s")
        console.print(f"Sample Rate: {info['sample_rate']} Hz")
        console.print(f"Channels: {info['channels']}")
        console.print(f"Codec: {info['codec']}")
        console.print(f"Frames: {info['frames']:,}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point."""
    app()
