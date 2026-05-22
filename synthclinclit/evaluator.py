"""Per-field evaluation of extraction predictions against a gold-standard CSV.

Two metric buckets:
  - paraphrase: sentence-embedding cosine similarity (handles wording variation,
    enum aliases, and the merged-boolean-string vs free-text gold column case).
  - numeric:    pull the first numeric token from both sides and compare exactly.

Field mapping is declared in a YAML/JSON config keyed by gold-column name:

    Study Design:
      from: [RCT, Observational, Pilot / Feasibility Study]
      metric: paraphrase
    Number of Participants:
      from: n_participants
      metric: numeric
    Health Literacy:
      from: Health Literacy
      metric: paraphrase
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Metric = Literal["paraphrase", "numeric"]


@dataclass
class FieldRule:
    gold_column: str
    from_columns: list[str]
    metric: Metric = "paraphrase"


@dataclass
class EvalConfig:
    rules: list[FieldRule]


@dataclass
class CellResult:
    source: str
    field: str
    metric: str
    gold: str
    predicted: str
    score: float | None  # None means not comparable (e.g. no number found on either side)


def load_eval_config(path: Path) -> EvalConfig:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError("pyyaml not installed; use JSON or `pip install pyyaml`") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    rules: list[FieldRule] = []
    for gold_col, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Config entry for {gold_col!r} must be a mapping")
        src = spec.get("from", gold_col)
        if isinstance(src, str):
            src = [src]
        rules.append(FieldRule(
            gold_column=gold_col,
            from_columns=list(src),
            metric=spec.get("metric", "paraphrase"),
        ))
    return EvalConfig(rules=rules)


def _read_csv_by_source(path: Path, source_col: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get(source_col) or "").strip()
            if key:
                out[key] = row
    return out


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(s: str) -> float | None:
    if not s:
        return None
    m = _NUMBER_RE.search(s.replace(",", ""))
    return float(m.group()) if m else None


def score_numeric(gold: str, pred: str) -> float | None:
    gn, pn = _extract_number(gold), _extract_number(pred)
    if gn is None or pn is None:
        return None
    return 1.0 if gn == pn else 0.0


def _resolve_predicted(pred_row: dict[str, str], from_columns: list[str]) -> str:
    parts = [
        v.strip() for c in from_columns
        for v in [pred_row.get(c) or ""] if v.strip()
    ]
    return "; ".join(parts)


def evaluate(
    predictions_path: Path,
    gold_path: Path,
    config: EvalConfig,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    source_col: str = "source",
) -> list[CellResult]:
    preds = _read_csv_by_source(predictions_path, source_col)
    gold = _read_csv_by_source(gold_path, source_col)

    rows: list[CellResult] = []
    paraphrase_tasks: list[tuple[int, str, str]] = []  # (index into rows, gold, pred)

    for src, gold_row in gold.items():
        pred_row = preds.get(src, {})
        for rule in config.rules:
            g_val = (gold_row.get(rule.gold_column) or "").strip()
            p_val = _resolve_predicted(pred_row, rule.from_columns)
            cell = CellResult(
                source=src, field=rule.gold_column, metric=rule.metric,
                gold=g_val, predicted=p_val, score=None,
            )
            if rule.metric == "numeric":
                cell.score = score_numeric(g_val, p_val)
            elif rule.metric == "paraphrase":
                if not g_val and not p_val:
                    cell.score = 1.0
                elif not g_val or not p_val:
                    cell.score = 0.0
                else:
                    paraphrase_tasks.append((len(rows), g_val, p_val))
            else:
                raise ValueError(f"Unknown metric: {rule.metric}")
            rows.append(cell)

    if paraphrase_tasks:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install 'synthclinclit[evaluate]' "
                "or pip install sentence-transformers"
            ) from e
        model = SentenceTransformer(embed_model_name)
        golds = [t[1] for t in paraphrase_tasks]
        preds_text = [t[2] for t in paraphrase_tasks]
        g_emb = model.encode(golds, convert_to_numpy=True, normalize_embeddings=True)
        p_emb = model.encode(preds_text, convert_to_numpy=True, normalize_embeddings=True)
        sims = (g_emb * p_emb).sum(axis=1).tolist()  # dot product == cosine for normalized
        for (idx, _, _), score in zip(paraphrase_tasks, sims):
            rows[idx].score = float(score)

    missing_in_pred = [s for s in gold if s not in preds]
    extra_in_pred = [s for s in preds if s not in gold]
    if missing_in_pred:
        print(f"[warn] {len(missing_in_pred)} gold source(s) had no matching prediction row "
              f"(first few: {missing_in_pred[:5]})")
    if extra_in_pred:
        print(f"[warn] {len(extra_in_pred)} prediction source(s) not present in gold "
              f"(first few: {extra_in_pred[:5]})")

    return rows
