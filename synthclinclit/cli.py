"""Typer-based CLI for end-to-end retrieval + extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .downloader import PMCDownloader
from .extractor import ArtefactExtractor, ArtefactSchema
from .id_resolver import resolve_ids
from .parser import parse_jats

app = typer.Typer(help="Download PMC full-text articles and extract user-defined artefacts.")
console = Console()


def _read_ids(ids: list[str] | None, id_file: Path | None) -> list[str]:
    collected: list[str] = []
    if ids:
        collected.extend(ids)
    if id_file:
        for line in id_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                collected.append(line)
    if not collected:
        raise typer.BadParameter("Provide at least one ID via --id or --id-file")
    return collected


@app.command()
def resolve(
    ids: list[str] = typer.Option(None, "--id", "-i", help="DOI, PMID, or PMCID (repeatable)"),
    id_file: Path = typer.Option(None, "--id-file", "-f", help="File with one ID per line"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
) -> None:
    """Resolve identifiers to DOI/PMID/PMCID triples."""
    raw = _read_ids(ids, id_file)
    results = resolve_ids(raw)
    if json_out:
        console.print_json(json.dumps([r.__dict__ for r in results]))
        return

    table = Table(title="Resolved IDs")
    for col in ("input", "type", "pmcid", "pmid", "doi", "error"):
        table.add_column(col)
    for r in results:
        table.add_row(
            r.input_id, r.input_type, r.pmcid or "", r.pmid or "", r.doi or "", r.error or ""
        )
    console.print(table)


@app.command()
def check(
    ids: list[str] = typer.Option(None, "--id", "-i"),
    id_file: Path = typer.Option(None, "--id-file", "-f"),
) -> None:
    """Check PMC Open Access availability for each article."""
    raw = _read_ids(ids, id_file)
    resolved = resolve_ids(raw)

    table = Table(title="PMC full-text availability")
    for col in ("input", "pmcid", "oa", "license", "reason"):
        table.add_column(col)

    with PMCDownloader() as dl:
        for r in resolved:
            if not r.pmcid:
                table.add_row(r.input_id, "", "-", "", r.error or "no PMCID")
                continue
            try:
                a = dl.check_availability(r.pmcid)
                table.add_row(r.input_id, a.pmcid, "yes" if a.oa_available else "no",
                              a.license or "", a.reason or "")
            except Exception as e:  # noqa: BLE001
                table.add_row(r.input_id, r.pmcid, "?", "", str(e))
    console.print(table)


@app.command()
def download(
    ids: list[str] = typer.Option(None, "--id", "-i"),
    id_file: Path = typer.Option(None, "--id-file", "-f"),
    out_dir: Path = typer.Option(Path("./articles"), "--out", "-o"),
    require_oa: bool = typer.Option(False, "--require-oa", help="Skip non-OA articles"),
) -> None:
    """Download JATS XML full text for each resolved article."""
    raw = _read_ids(ids, id_file)
    resolved = resolve_ids(raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title=f"Downloads → {out_dir}")
    for col in ("input", "pmcid", "status", "path"):
        table.add_column(col)

    with PMCDownloader() as dl:
        for r in resolved:
            if not r.pmcid:
                table.add_row(r.input_id, "", "skip", r.error or "no PMCID")
                continue
            try:
                path = dl.download(r, out_dir, require_oa=require_oa)
                if path is None:
                    table.add_row(r.input_id, r.pmcid, "skip", "not OA or unavailable")
                else:
                    table.add_row(r.input_id, r.pmcid, "ok", str(path))
            except Exception as e:  # noqa: BLE001
                table.add_row(r.input_id, r.pmcid, "error", str(e))
    console.print(table)


@app.command()
def extract(
    schema_path: Path = typer.Option(..., "--schema", "-s", help="Artefact schema JSON/YAML"),
    ids: list[str] = typer.Option(None, "--id", "-i"),
    id_file: Path = typer.Option(None, "--id-file", "-f"),
    xml_dir: Path = typer.Option(None, "--xml-dir", help="Use already-downloaded XML from this directory"),
    out_dir: Path = typer.Option(Path("./articles"), "--out", "-o"),
    backend: str = typer.Option("anthropic", "--backend", "-b",
                                help="LLM backend: 'anthropic' (hosted) or 'ollama' (local)"),
    model: str = typer.Option(None, "--model", "-m",
                              help="Model name. Defaults: anthropic=claude-opus-4-7, ollama=llama3.1:8b"),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host",
                                    help="Ollama server URL (only used when --backend=ollama)"),
    num_ctx: int = typer.Option(8192, "--num-ctx",
                                help="Ollama context window size; raise for long articles"),
    results_path: Path = typer.Option(Path("./extractions.json"), "--results", "-r"),
) -> None:
    """Download (if needed), parse, then run LLM extraction using the user schema."""
    schema = ArtefactSchema.from_file(schema_path)

    articles = []
    if xml_dir and ids is None and id_file is None:
        xml_files = sorted(xml_dir.glob("*.xml"))
    else:
        raw = _read_ids(ids, id_file)
        resolved = resolve_ids(raw)
        xml_files = []
        with PMCDownloader() as dl:
            for r in resolved:
                if not r.pmcid:
                    console.print(f"[yellow]skip[/] {r.input_id}: {r.error or 'no PMCID'}")
                    continue
                try:
                    path = dl.download(r, out_dir)
                    if path is None:
                        console.print(f"[yellow]skip[/] {r.input_id}: full text unavailable")
                        continue
                    xml_files.append(path)
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]error[/] {r.input_id}: {e}")

    for p in xml_files:
        try:
            articles.append(parse_jats(p))
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]parse error[/] {p.name}: {e}")

    if not articles:
        console.print("[red]No articles available for extraction.[/]")
        raise typer.Exit(code=1)

    backend_kwargs: dict = {}
    if backend == "ollama":
        backend_kwargs = {"host": ollama_host, "num_ctx": num_ctx}
    extractor = ArtefactExtractor(
        schema=schema, backend=backend, model=model, **backend_kwargs
    )
    console.print(f"[cyan]extracting with[/] backend={backend} model={extractor.backend.model}")
    results = extractor.extract(articles)

    payload = [
        {"source": r.source, "data": r.data, "errors": r.errors, "raw": r.raw}
        for r in results
    ]
    results_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]Wrote {len(payload)} extraction(s) →[/] {results_path}")

    if sys.stdout.isatty():
        console.print_json(json.dumps([{"source": r.source, "data": r.data} for r in results]))


if __name__ == "__main__":
    app()
