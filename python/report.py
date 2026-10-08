"""Terminal rendering and result logs, shared by run_sim.py and sweep.py.

Presentation only: nothing here computes or changes a result. A table is a
list of rows of cells; a cell is (text, rich style). The terminal gets the
styles, the Markdown log gets the plain text, so both always show the same
numbers.

Colors carry one meaning each:
  green   within the target (no off-track step, ESS p5 in the 1-10 % of K band)
  yellow  close to a limit (|d| > 95 % of d_max, tire past its linear zone,
          ESS p5 above 10 % of K: safe but timid)
  red     failure (off track, lap not done, ESS p5 < 1 % of K, a_lat > mu g)
"""
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

console = Console(highlight=False)

Cell = tuple[str, str]   # (texte, style rich) ; le style n'apparaît pas dans les logs
Row = list[Cell] | None  # None : séparateur entre deux groupes de lignes

OK, WARN, BAD = "green", "yellow", "bold red"


# --- En-têtes et progression ---------------------------------------------------

def header(title: str, details: str = "") -> None:
    """One line in the pixi style: '✨ title  details'."""
    console.print(f"[bold magenta]✨ {escape(title)}[/]  [dim]{escape(details)}[/]")


def done(message: str) -> None:
    console.print(f"[green]✔[/] {message}")


def progress() -> Progress:
    """Spinner, bar and elapsed time. Use as a context manager."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        TextColumn("{task.fields[info]}", style="dim"),
        TimeElapsedColumn(),
        console=console,
    )


def job_progress() -> Progress:
    """Same look, with an 'n/N' counter, for a pool of independent runs."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


# --- Cellules notées -------------------------------------------------------------

def cell(text: str, style: str = "") -> Cell:
    return text, style


def lap_cell(r: dict) -> Cell:
    if r["lap"]:
        return cell(f"✔ {r['t']:.2f} s", OK if r["off"] == 0 else WARN)
    return cell(f"✘ {r['progress']:.1f}/{r['length']:.1f} m", BAD)


def off_cell(n: int) -> Cell:
    return cell(str(n), OK if n == 0 else BAD)


def dmax_cell(d: float, limit: float) -> Cell:
    style = BAD if d > limit else WARN if d > 0.95 * limit else ""
    return cell(f"{d:.2f}", style)


def ess_cell(med: float, p5: float, k: int) -> Cell:
    """Theory 4.6: ESS p5 between 1 % and 10 % of K."""
    style = BAD if p5 < 0.01 * k else OK if p5 <= 0.1 * k else WARN
    return cell(f"{med:.0f} / {p5:.0f}", style)


def alpha_cell(deg: float, sat_deg: float) -> Cell:
    """Past 1/C_S the tire leaves its linear zone."""
    return cell(f"{deg:.1f}", WARN if deg > sat_deg else "")


def clean_cell(clean: int, n: int) -> Cell:
    return cell(f"{clean}/{n}", OK if clean == n else BAD if clean == 0 else WARN)


# --- Plusieurs graines -----------------------------------------------------------

SEEDS_COLUMNS = ["variant", "clean", "off per seed", "lap (s)", "|d|", "ESS med"]
SEEDS_CAPTION = ("clean: full lap with no off-track step · off per seed: off-track steps · "
                 "|d|: max lateral offset over the seeds (m)")


def seeds_row(name: str, rs: list[dict], style: str = "") -> Row:
    """One variant over several seeds. A clean lap is a full lap with off = 0."""
    clean = sum(r["lap"] and r["off"] == 0 for r in rs)
    t = [r["t"] for r in rs]
    ess = [r["ess_med"] for r in rs]
    return [
        cell(name, style),
        clean_cell(clean, len(rs)),
        cell(str([r["off"] for r in rs]), OK if clean == len(rs) else ""),
        cell(f"{min(t):.2f} to {max(t):.2f}"),
        dmax_cell(max(r["dmax"] for r in rs), rs[0]["d_limit"]),
        cell(f"{min(ess):.0f} to {max(ess):.0f}"),
    ]


# --- Tableaux --------------------------------------------------------------------

def print_table(columns: list[str], rows: list[Row], caption: str = "") -> None:
    """Numbers never wrap; only the first column (labels) folds on a narrow terminal."""
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", caption=caption or None,
                  caption_style="dim", caption_justify="left", pad_edge=False, collapse_padding=True)
    for i, name in enumerate(columns):
        if i == 0:
            table.add_column(name, min_width=12, overflow="fold")
        else:
            table.add_column(name, justify="right", no_wrap=True)
    for row in rows:
        if row is None:
            table.add_section()
        else:
            table.add_row(*(f"[{s}]{escape(t)}[/]" if s else escape(t) for t, s in row))
    console.print(table)


def markdown_table(columns: list[str], rows: list[Row], caption: str = "") -> str:
    esc = lambda t: t.replace("|", "\\|")   # noqa: E731  ("|d|" casserait les colonnes)
    lines = ["| " + " | ".join(map(esc, columns)) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(esc(t) for t, _ in row) + " |" for row in rows if row is not None]
    return "\n".join(lines) + (f"\n\n{caption}" if caption else "")


# --- Logs ------------------------------------------------------------------------

def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_log(results: Path, kind: str, tag: str, markdown: str, data: dict) -> Path:
    """results/logs/<kind>_<date>_<tag>.md (readable) and .json (full config, raw metrics).

    Returns the .md path. The command line is recorded in both.
    """
    stamp = datetime.now()
    stem = results / "logs" / f"{kind}_{stamp:%Y-%m-%d_%H-%M-%S}_{tag}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    command = "python " + " ".join(sys.argv)
    md = f"# {kind} {stamp:%Y-%m-%d %H:%M:%S}\n\n`{command}`\n\n{markdown}\n"
    stem.with_suffix(".md").write_text(md)
    payload = {"date": stamp.isoformat(timespec="seconds"), "command": command, **data}
    stem.with_suffix(".json").write_text(json.dumps(_jsonable(payload), indent=1))
    return stem.with_suffix(".md")


def relative(path: Path) -> str:
    """Path relative to the working directory when possible, for short output."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)
