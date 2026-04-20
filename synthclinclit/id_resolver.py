"""Resolve between DOI, PMID, and PMCID using NCBI's ID Converter."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

import httpx
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import IDCONV_BASE, NCBI

IDType = Literal["doi", "pmid", "pmcid", "unknown"]

_PMCID_RE = re.compile(r"^(PMC)?\d+$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d+$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


@dataclass
class ArticleIDs:
    """Normalized set of identifiers for one article."""
    input_id: str
    input_type: IDType
    pmcid: str | None = None
    pmid: str | None = None
    doi: str | None = None
    error: str | None = None

    @property
    def resolved(self) -> bool:
        return self.error is None and self.pmcid is not None


def classify_id(raw: str) -> IDType:
    s = raw.strip()
    if not s:
        return "unknown"
    if s.lower().startswith("pmc") and _PMCID_RE.match(s):
        return "pmcid"
    if _DOI_RE.match(s):
        return "doi"
    if _PMID_RE.match(s):
        # Ambiguous: a bare number could be a PMID or a PMCID.
        # NCBI convention: PMCIDs carry the PMC prefix; a bare integer is a PMID.
        return "pmid"
    return "unknown"


def normalize_pmcid(value: str) -> str:
    v = value.strip().upper()
    return v if v.startswith("PMC") else f"PMC{v}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5))
def _call_idconv(client: httpx.Client, ids: list[str]) -> bytes:
    params = {
        "ids": ",".join(ids),
        "format": "xml",
        "versions": "no",
        **NCBI.common_params(),
    }
    r = client.get(IDCONV_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.content


def resolve_ids(
    raw_ids: Iterable[str],
    client: httpx.Client | None = None,
    batch_size: int = 100,
) -> list[ArticleIDs]:
    """Resolve a batch of mixed identifiers to DOI/PMID/PMCID triples.

    The NCBI ID Converter accepts mixed ID types in one call and returns
    whichever cross-references are known. Inputs that fail classification
    are returned with an error set, untouched.
    """
    ids_list = [s.strip() for s in raw_ids if s and s.strip()]
    results: dict[str, ArticleIDs] = {}

    resolvable: list[str] = []
    for raw in ids_list:
        kind = classify_id(raw)
        if kind == "unknown":
            results[raw] = ArticleIDs(raw, kind, error="unrecognized id format")
            continue
        results[raw] = ArticleIDs(raw, kind)
        resolvable.append(raw)

    owns_client = client is None
    client = client or httpx.Client()
    try:
        for i in range(0, len(resolvable), batch_size):
            chunk = resolvable[i : i + batch_size]
            try:
                xml_bytes = _call_idconv(client, chunk)
            except Exception as exc:  # noqa: BLE001
                for r in chunk:
                    results[r].error = f"idconv failed: {exc}"
                continue

            root = etree.fromstring(xml_bytes)
            for rec in root.findall("record"):
                status = rec.get("status")
                req = rec.get("requested-id") or ""
                target = results.get(req)
                if target is None:
                    continue
                if status == "error":
                    target.error = rec.get("errmsg") or "idconv error"
                    continue
                pmcid = rec.get("pmcid")
                if pmcid:
                    target.pmcid = normalize_pmcid(pmcid)
                pmid = rec.get("pmid")
                if pmid:
                    target.pmid = pmid
                doi = rec.get("doi")
                if doi:
                    target.doi = doi
    finally:
        if owns_client:
            client.close()

    return [results[r] for r in ids_list]
