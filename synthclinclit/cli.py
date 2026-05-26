"""Typer-based CLI for end-to-end retrieval + extraction."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .downloader import PMCDownloader
from .extractor import ArtefactExtractor, ArtefactSchema
from .id_resolver import resolve_ids
from .parser import ParsedArticle, parse_jats

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
    text_dir: Path = typer.Option(None, "--text-dir",
                                  help="Use plain-text articles (one .txt per article) from this directory. "
                                       "Filename stem is used as the 'source' in the output."),
    out_dir: Path = typer.Option(Path("./articles"), "--out", "-o"),
    backend: str = typer.Option("anthropic", "--backend", "-b",
                                help="LLM backend: 'anthropic' (hosted) or 'ollama' (local)"),
    model: str = typer.Option(None, "--model", "-m",
                              help="Model name. Defaults: anthropic=claude-opus-4-7, ollama=llama3.1:8b"),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host",
                                    help="Ollama server URL (only used when --backend=ollama)"),
    num_ctx: int = typer.Option(8192, "--num-ctx",
                                help="Ollama context window size; raise for long articles"),
    ollama_timeout: float = typer.Option(1800.0, "--ollama-timeout",
                                         help="HTTP read timeout (seconds) for Ollama /api/chat. "
                                              "Raise if long articles + large num_ctx hit ReadTimeout."),
    results_path: Path = typer.Option(Path("./extractions.json"), "--results", "-r"),
) -> None:
    """Download (if needed), parse, then run LLM extraction using the user schema."""
    schema = ArtefactSchema.from_file(schema_path)

    if text_dir and (xml_dir or ids or id_file):
        raise typer.BadParameter("--text-dir is mutually exclusive with --xml-dir / --id / --id-file")

    articles = []
    xml_files: list[Path] = []
    if text_dir:
        for p in sorted(text_dir.glob("*.txt")):
            try:
                text = p.read_text().strip()
                if not text:
                    console.print(f"[yellow]skip[/] {p.name}: empty file")
                    continue
                articles.append(ParsedArticle(source=p.stem, sections=[("", text)]))
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]read error[/] {p.name}: {e}")
    elif xml_dir and ids is None and id_file is None:
        xml_files = sorted(xml_dir.glob("*.xml"))
    else:
        raw = _read_ids(ids, id_file)
        resolved = resolve_ids(raw)
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
        backend_kwargs = {"host": ollama_host, "num_ctx": num_ctx, "timeout": ollama_timeout}
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


@app.command("to-csv")
def to_csv(
    results_path: Path = typer.Option(Path("./extractions.json"), "--results", "-r",
                                      help="Extraction results JSON produced by `extract`"),
    out_path: Path = typer.Option(Path("./extractions.csv"), "--out", "-o",
                                  help="Destination CSV path"),
    schema_path: Path = typer.Option(None, "--schema", "-s",
                                     help="Optional schema for deterministic column order"),
    keep_empty: bool = typer.Option(False, "--keep-empty",
                                    help="Keep columns whose every cell is null/false/empty "
                                         "(by default such columns are dropped)"),
    merge_booleans: str = typer.Option(None, "--merge-booleans",
                                       help="Collapse every boolean column into a single column "
                                            "with this name. Each row's cell lists ('; '-joined) "
                                            "the names of the boolean fields that were true."),
) -> None:
    """Convert extraction results JSON to CSV.

    Per-cell rules (only populated fields produce content):
      - boolean true  → the field name (e.g. 'Reports Intervention')
      - boolean false → empty
      - string value  → the value itself
      - list[string]  → values joined with '; '
      - null / empty  → empty

    Columns whose every cell ends up empty are dropped unless --keep-empty.
    With --merge-booleans NAME, all boolean fields are folded into one
    column under NAME before the empty-column pass.
    """
    payload = json.loads(results_path.read_text())

    schema = ArtefactSchema.from_file(schema_path) if schema_path else None
    if schema:
        columns = [f.name for f in schema.fields]
    else:
        columns = []
        for row in payload:
            for k in (row.get("data") or {}):
                if k not in columns:
                    columns.append(k)

    def cell(name: str, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return name if value else ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "; ".join(str(x) for x in value if x not in (None, ""))
        return str(value)

    matrix = [
        [cell(c, (row.get("data") or {}).get(c)) for c in columns]
        for row in payload
    ]

    if merge_booleans:
        if schema:
            bool_cols = {f.name for f in schema.fields if f.type == "boolean"}
        else:
            bool_cols = set()
            for c in columns:
                seen = False
                ok = True
                for row in payload:
                    v = (row.get("data") or {}).get(c)
                    if v is None:
                        continue
                    if not isinstance(v, bool):
                        ok = False
                        break
                    seen = True
                if seen and ok:
                    bool_cols.add(c)

        if bool_cols:
            first_bool_idx = next((i for i, c in enumerate(columns) if c in bool_cols), len(columns))
            insert_at = sum(1 for c in columns[:first_bool_idx] if c not in bool_cols)

            merged_cells = [
                "; ".join(c for c, v in zip(columns, row_vals) if c in bool_cols and v == c)
                for row_vals in matrix
            ]

            keep_mask = [c not in bool_cols for c in columns]
            columns = [c for c, k in zip(columns, keep_mask) if k]
            matrix = [[v for v, k in zip(row, keep_mask) if k] for row in matrix]

            columns.insert(insert_at, merge_booleans)
            for row, m in zip(matrix, merged_cells):
                row.insert(insert_at, m)

    if not keep_empty and columns:
        keep = [any(matrix[r][i] != "" for r in range(len(matrix))) for i in range(len(columns))]
        columns = [c for c, k in zip(columns, keep) if k]
        matrix = [[v for v, k in zip(row, keep) if k] for row in matrix]

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source"] + columns)
        for row, values in zip(payload, matrix):
            writer.writerow([row.get("source", "")] + values)

    console.print(f"[green]Wrote CSV →[/] {out_path}")


@app.command("evaluate")
def evaluate_cmd(
    predictions_path: Path = typer.Option(..., "--predictions", "-p",
                                          help="Predictions CSV (from `to-csv`)"),
    gold_path: Path = typer.Option(..., "--gold", "-g", help="Gold-standard CSV"),
    config_path: Path = typer.Option(..., "--config", "-c",
                                     help="YAML/JSON mapping config (gold_column → from / metric)"),
    out_path: Path = typer.Option(Path("./evaluation.csv"), "--out", "-o",
                                  help="Per-cell results CSV"),
    summary_path: Path = typer.Option(None, "--summary",
                                      help="Optional per-field summary JSON"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embed-model",
                                    help="Sentence-transformers model id"),
    source_col: str = typer.Option("source", "--source-col",
                                   help="Column used to join predictions and gold rows"),
) -> None:
    """Score predictions against a gold-standard CSV using a per-field policy.

    Each gold column maps to one or more predicted columns. Two metrics:
      - paraphrase: sentence-embedding cosine similarity (handles wording/aliases).
      - numeric:    first number extracted from each side, exact compare.
    """
    from .evaluator import evaluate as _evaluate, load_eval_config

    cfg = load_eval_config(config_path)
    console.print(f"[cyan]Evaluating[/] {len(cfg.rules)} field(s) using embed_model={embed_model}")
    results = _evaluate(
        predictions_path=predictions_path,
        gold_path=gold_path,
        config=cfg,
        embed_model_name=embed_model,
        source_col=source_col,
    )

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "field", "metric", "score", "gold", "predicted"])
        for r in results:
            writer.writerow([
                r.source, r.field, r.metric,
                "" if r.score is None else f"{r.score:.4f}",
                r.gold, r.predicted,
            ])
    console.print(f"[green]Wrote per-cell scores →[/] {out_path}")

    from collections import defaultdict
    per_field: dict[str, list[float]] = defaultdict(list)
    field_metric: dict[str, str] = {}
    for r in results:
        field_metric.setdefault(r.field, r.metric)
        if r.score is not None:
            per_field[r.field].append(r.score)

    table = Table(title="Per-field mean score")
    for col in ("field", "metric", "n", "mean"):
        table.add_column(col)
    overall: list[float] = []
    for fld, scores in per_field.items():
        m = sum(scores) / len(scores) if scores else 0.0
        table.add_row(fld, field_metric[fld], str(len(scores)), f"{m:.3f}")
        overall.extend(scores)
    if overall:
        table.add_row("ALL", "-", str(len(overall)), f"{sum(overall)/len(overall):.3f}")
    console.print(table)

    if summary_path:
        summary = {
            "per_field": {
                fld: {"metric": field_metric[fld],
                      "n": len(scores),
                      "mean": sum(scores) / len(scores) if scores else None}
                for fld, scores in per_field.items()
            },
            "overall": {"n": len(overall),
                        "mean": sum(overall) / len(overall) if overall else None},
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        console.print(f"[green]Wrote summary →[/] {summary_path}")


if __name__ == "__main__":
    app()
