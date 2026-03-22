#!/usr/bin/env python3
"""
Tamil Audio -> English Transcript CLI Tool
Converts Tamil audio/video files into English text transcripts
using faster-whisper (local GPU) or OpenAI Whisper API (cloud).
"""

import os
import sys
import time

# Fix Windows console encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console(force_terminal=True)

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac",
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv",
}

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]


def validate_input_file(file_path: Path) -> None:
    """Validate that the input file exists and is a supported format."""
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        sys.exit(1)
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(f"[red]Error:[/red] Unsupported format: {file_path.suffix}")
        console.print(f"[yellow]Supported formats:[/yellow] {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)


def get_output_path(input_path: Path, output_path: str | None) -> Path:
    """Determine the output file path. Saves to sibling 'transcripts' folder if it exists."""
    if output_path:
        return Path(output_path)
    # Check if a 'transcripts' sibling folder exists (next to 'source' folder)
    transcripts_dir = input_path.parent.parent / "transcripts"
    if not transcripts_dir.exists():
        # Also check if input is directly in a folder with a 'transcripts' subfolder
        transcripts_dir = input_path.parent / "transcripts"
    if transcripts_dir.exists() and transcripts_dir.is_dir():
        return transcripts_dir / f"{input_path.stem}_english.txt"
    return input_path.with_name(f"{input_path.stem}_english.txt")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_local(
    input_file: Path,
    model_size: str,
    include_timestamps: bool,
    language: str,
    verbose: bool,
) -> str:
    """Process audio locally using faster-whisper on GPU."""
    # Add NVIDIA CUDA library paths for GPU support (pip-installed nvidia packages)
    try:
        import nvidia.cublas
        import nvidia.cudnn
        for pkg in [nvidia.cublas, nvidia.cudnn]:
            for base_path in pkg.__path__:
                bin_path = os.path.join(base_path, "bin")
                lib_path = os.path.join(base_path, "lib")
                for p in [bin_path, lib_path]:
                    if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass  # CUDA pip packages not installed, will fall back to system CUDA or CPU

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        console.print("[red]Error:[/red] faster-whisper is not installed.")
        console.print("Run: [cyan]pip install faster-whisper[/cyan]")
        sys.exit(1)

    # Detect device — try ctranslate2 first (more reliable), then torch
    device = "cpu"
    compute_type = "int8"

    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in cuda_types:
            device = "cuda"
            compute_type = "float16"
            console.print("[green]GPU detected![/green] Using CUDA with float16.")
        elif cuda_types:
            device = "cuda"
            compute_type = "int8"
            console.print("[green]GPU detected![/green] Using CUDA with int8.")
        else:
            console.print("[yellow]Note:[/yellow] No GPU detected. Using CPU (slower).")
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                console.print("[green]GPU detected![/green] Using CUDA for acceleration.")
            else:
                console.print("[yellow]Warning:[/yellow] CUDA not available. Using CPU (slower).")
        except ImportError:
            console.print("[yellow]Note:[/yellow] No GPU detected. Using CPU (slower).")

    if verbose:
        console.print(f"[dim]Device: {device} | Compute type: {compute_type}[/dim]")

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(f"Loading {model_size} model on {device.upper()}...", total=None)
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                console.print("[yellow]Warning:[/yellow] CUDA failed. Falling back to CPU.")
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
            else:
                console.print(f"[red]Error loading model:[/red] {e}")
                sys.exit(1)

    # Transcribe + translate
    console.print(f"[cyan]Processing:[/cyan] {input_file.name}")
    console.print(f"[dim]Model: {model_size} | Task: translate (Tamil -> English)[/dim]")

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing and translating...", total=None)

        segments_list = []
        try:
            segments, info = model.transcribe(
                str(input_file),
                task="translate",
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out silence for better results
            )

            if verbose:
                console.print(f"[dim]Detected language: {info.language} (probability: {info.language_probability:.2f})[/dim]")
                console.print(f"[dim]Audio duration: {info.duration:.1f}s[/dim]")

            for segment in segments:
                segments_list.append(segment)
                progress.update(task, description=f"Translating... [{format_timestamp(segment.end)}]")

        except Exception as e:
            console.print(f"[red]Error during transcription:[/red] {e}")
            if "out of memory" in str(e).lower():
                console.print("[yellow]Tip:[/yellow] Try a smaller model with [cyan]-m small[/cyan] or use [cyan]--use-api[/cyan]")
            sys.exit(1)

    elapsed_time = time.time() - start_time

    # Format output
    output_lines = []
    for segment in segments_list:
        text = segment.text.strip()
        if not text:
            continue
        if include_timestamps:
            start_ts = format_timestamp(segment.start)
            end_ts = format_timestamp(segment.end)
            output_lines.append(f"[{start_ts} -> {end_ts}] {text}")
        else:
            output_lines.append(text)

    transcript = "\n".join(output_lines)

    # Print summary
    word_count = len(transcript.split())
    audio_duration = info.duration if info else 0

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_row("[dim]Audio duration:[/dim]", f"{audio_duration:.1f}s")
    summary_table.add_row("[dim]Processing time:[/dim]", f"{elapsed_time:.1f}s")
    summary_table.add_row("[dim]Speed:[/dim]", f"{audio_duration / elapsed_time:.1f}x realtime" if elapsed_time > 0 else "N/A")
    summary_table.add_row("[dim]Word count:[/dim]", str(word_count))
    console.print(Panel(summary_table, title="Summary", border_style="green"))

    return transcript


def process_api(
    input_file: Path,
    include_timestamps: bool,
    verbose: bool,
) -> str:
    """Process audio using OpenAI Whisper API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY not found.")
        console.print("Set it in a [cyan].env[/cyan] file or as an environment variable.")
        console.print("See [cyan].env.example[/cyan] for the template.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        console.print("[red]Error:[/red] openai package is not installed.")
        console.print("Run: [cyan]pip install openai[/cyan]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Check file size (API limit is 25MB)
    file_size_mb = input_file.stat().st_size / (1024 * 1024)
    if file_size_mb > 25:
        console.print(f"[red]Error:[/red] File is {file_size_mb:.1f}MB. OpenAI API limit is 25MB.")
        console.print("[yellow]Tip:[/yellow] Use local mode (without --use-api) for large files.")
        sys.exit(1)

    console.print(f"[cyan]Processing via API:[/cyan] {input_file.name} ({file_size_mb:.1f}MB)")

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
        console=console,
    ) as progress:
        progress.add_task("Uploading and translating via OpenAI API...", total=None)

        try:
            with open(input_file, "rb") as audio_file:
                if include_timestamps:
                    # Use verbose_json to get timestamps
                    response = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )
                    lines = []
                    for segment in response.segments:
                        seg_start = segment.start if hasattr(segment, "start") else segment["start"]
                        seg_end = segment.end if hasattr(segment, "end") else segment["end"]
                        seg_text = segment.text if hasattr(segment, "text") else segment["text"]
                        start_ts = format_timestamp(seg_start)
                        end_ts = format_timestamp(seg_end)
                        text = seg_text.strip()
                        if text:
                            lines.append(f"[{start_ts} -> {end_ts}] {text}")
                    transcript = "\n".join(lines)
                else:
                    response = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                    )
                    transcript = response.strip() if isinstance(response, str) else response.text.strip()

        except Exception as e:
            console.print(f"[red]API Error:[/red] {e}")
            sys.exit(1)

    elapsed_time = time.time() - start_time
    word_count = len(transcript.split())

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_row("[dim]Processing time:[/dim]", f"{elapsed_time:.1f}s")
    summary_table.add_row("[dim]Word count:[/dim]", str(word_count))
    summary_table.add_row("[dim]Cost (approx):[/dim]", f"~${file_size_mb * 0.006:.4f}")
    console.print(Panel(summary_table, title="API Summary", border_style="blue"))

    return transcript


def get_audio_files(folder: Path) -> list[Path]:
    """Get all supported audio/video files from a folder."""
    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


@click.command()
@click.argument("input_file", required=False, type=click.Path())
@click.option("--output", "-o", type=click.Path(), help="Output .txt file path")
@click.option("--model", "-m", type=click.Choice(MODEL_SIZES), default="medium", help="Whisper model size (default: medium)")
@click.option("--timestamps/--no-timestamps", default=False, help="Include timestamps in output")
@click.option("--use-api", is_flag=True, default=False, help="Use OpenAI Whisper API instead of local GPU")
@click.option("--language", "-l", default="ta", help="Source language code (default: ta for Tamil)")
@click.option("--batch", "-b", type=click.Path(exists=True), help="Process all audio files in a folder")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed processing info")
def main(input_file, output, model, timestamps, use_api, language, batch, verbose):
    """
    Convert Tamil audio/video files to English text transcripts.

    Uses faster-whisper (local GPU) by default, or OpenAI Whisper API with --use-api.

    \b
    Examples:
      python tamil_to_english.py recording.mp3
      python tamil_to_english.py recording.mp3 -m large-v3
      python tamil_to_english.py recording.mp3 --timestamps
      python tamil_to_english.py recording.mp3 --use-api
      python tamil_to_english.py --batch ./my_audio_files/
    """
    console.print(Panel(
        "[bold cyan]Tamil Audio -> English Transcript[/bold cyan]",
        subtitle="faster-whisper | OpenAI API",
        border_style="cyan",
    ))

    # Batch mode
    if batch:
        batch_folder = Path(batch)
        audio_files = get_audio_files(batch_folder)
        if not audio_files:
            console.print(f"[red]No supported audio files found in:[/red] {batch_folder}")
            sys.exit(1)

        console.print(f"[cyan]Found {len(audio_files)} file(s) to process[/cyan]\n")

        for i, file_path in enumerate(audio_files, 1):
            console.print(f"\n[bold]── File {i}/{len(audio_files)} ──[/bold]")
            output_path = get_output_path(file_path, None)

            if use_api:
                transcript = process_api(file_path, timestamps, verbose)
            else:
                transcript = process_local(file_path, model, timestamps, language, verbose)

            output_path.write_text(transcript, encoding="utf-8")
            console.print(f"[green]Saved:[/green] {output_path}")

        console.print(f"\n[bold green]Batch complete![/bold green] Processed {len(audio_files)} file(s).")
        return

    # Single file mode
    if not input_file:
        console.print("[red]Error:[/red] Please provide an input file or use --batch.")
        console.print("Run [cyan]python tamil_to_english.py --help[/cyan] for usage.")
        sys.exit(1)

    file_path = Path(input_file)
    validate_input_file(file_path)
    output_path = get_output_path(file_path, output)

    if use_api:
        transcript = process_api(file_path, timestamps, verbose)
    else:
        transcript = process_local(file_path, model, timestamps, language, verbose)

    # Save output
    output_path.write_text(transcript, encoding="utf-8")
    console.print(f"\n[bold green]Transcript saved:[/bold green] {output_path}")

    # Preview first few lines
    lines = transcript.split("\n")
    preview_count = min(5, len(lines))
    if preview_count > 0:
        console.print(f"\n[dim]── Preview (first {preview_count} lines) ──[/dim]")
        for line in lines[:preview_count]:
            console.print(f"  {line}")
        if len(lines) > preview_count:
            console.print(f"  [dim]... and {len(lines) - preview_count} more lines[/dim]")


if __name__ == "__main__":
    main()
