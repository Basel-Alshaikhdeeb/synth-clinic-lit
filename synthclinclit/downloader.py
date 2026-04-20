"""Check PMC open-access availability and download JATS XML full text."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import EUTILS_BASE, NCBI, OA_BASE
from .id_resolver import ArticleIDs, normalize_pmcid


@dataclass
class FullTextAvailability:
    pmcid: str
    oa_available: bool
    license: str | None = None
    xml_url: str | None = None
    pdf_url: str | None = None
    reason: str | None = None


class PMCDownloader:
    """Fetches full-text XML from PubMed Central.

    PMC exposes two relevant endpoints:
      - OA service: indicates whether an article is in the Open Access subset
        and returns direct package URLs (XML, PDF).
      - EFetch: returns JATS XML for any PMC record (OA or not), but access
        may be restricted for non-OA articles.
    """

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=60, follow_redirects=True)
        self._owns_client = client is None

    def __enter__(self) -> "PMCDownloader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5))
    def check_availability(self, pmcid: str) -> FullTextAvailability:
        pmcid = normalize_pmcid(pmcid)
        params = {"id": pmcid, **NCBI.common_params()}
        r = self._client.get(OA_BASE, params=params)
        r.raise_for_status()
        root = etree.fromstring(r.content)

        err = root.find("error")
        if err is not None:
            return FullTextAvailability(
                pmcid=pmcid,
                oa_available=False,
                reason=err.get("code") or (err.text or "").strip() or "not in OA subset",
            )

        record = root.find(".//record")
        if record is None:
            return FullTextAvailability(pmcid=pmcid, oa_available=False, reason="no record")

        xml_url = pdf_url = None
        for link in record.findall("link"):
            fmt = link.get("format")
            href = link.get("href")
            if fmt == "tgz" and href and xml_url is None:
                # Package tarballs use FTP; we prefer direct XML via efetch below.
                xml_url = href
            elif fmt == "pdf" and href:
                pdf_url = href

        return FullTextAvailability(
            pmcid=pmcid,
            oa_available=True,
            license=record.get("license"),
            xml_url=xml_url,
            pdf_url=pdf_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5))
    def fetch_jats_xml(self, pmcid: str) -> bytes:
        """Retrieve JATS XML via EFetch. Works for most OA articles."""
        pmcid = normalize_pmcid(pmcid)
        numeric = pmcid.removeprefix("PMC")
        params = {
            "db": "pmc",
            "id": numeric,
            "rettype": "xml",
            "retmode": "xml",
            **NCBI.common_params(),
        }
        r = self._client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        r.raise_for_status()
        if b"<article" not in r.content:
            raise RuntimeError(f"EFetch returned no <article> for {pmcid}")
        return r.content

    def download(
        self,
        ids: ArticleIDs,
        out_dir: Path,
        require_oa: bool = False,
    ) -> Path | None:
        """Download JATS XML for an article. Returns the written path or None."""
        if not ids.pmcid:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)

        avail = self.check_availability(ids.pmcid)
        if require_oa and not avail.oa_available:
            return None

        try:
            xml_bytes = self.fetch_jats_xml(ids.pmcid)
        except Exception:
            if require_oa:
                raise
            return None

        path = out_dir / f"{ids.pmcid}.xml"
        path.write_bytes(xml_bytes)
        return path
