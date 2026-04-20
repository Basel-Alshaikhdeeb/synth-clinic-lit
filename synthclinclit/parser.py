"""Parse JATS XML full-text into structured plain sections for LLM input."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree


@dataclass
class ParsedArticle:
    pmcid: str | None = None
    doi: str | None = None
    pmid: str | None = None
    title: str = ""
    journal: str = ""
    year: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    sections: list[tuple[str, str]] = field(default_factory=list)  # (heading, text)

    def as_text(self, include_abstract: bool = True) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"# {self.title}")
        header_bits = []
        if self.journal:
            header_bits.append(self.journal)
        if self.year:
            header_bits.append(self.year)
        if self.authors:
            header_bits.append(", ".join(self.authors[:6]) + (" et al." if len(self.authors) > 6 else ""))
        if header_bits:
            parts.append(" | ".join(header_bits))
        if include_abstract and self.abstract:
            parts.append("## Abstract\n" + self.abstract)
        for heading, text in self.sections:
            parts.append(f"## {heading}\n{text}" if heading else text)
        return "\n\n".join(parts).strip()


def _text_of(elem: etree._Element | None) -> str:
    if elem is None:
        return ""
    return " ".join(elem.itertext()).strip()


def _find_meta(root: etree._Element, xpath: str) -> str:
    node = root.find(xpath)
    return _text_of(node) if node is not None else ""


def _extract_ids(root: etree._Element) -> tuple[str | None, str | None, str | None]:
    pmcid = pmid = doi = None
    for aid in root.findall(".//article-meta/article-id"):
        kind = aid.get("pub-id-type", "")
        val = (aid.text or "").strip()
        if not val:
            continue
        if kind == "pmc":
            pmcid = f"PMC{val.lstrip('PMC')}"
        elif kind == "pmid":
            pmid = val
        elif kind == "doi":
            doi = val
    return pmcid, pmid, doi


def _extract_year(root: etree._Element) -> str | None:
    for xp in (
        ".//article-meta/pub-date[@pub-type='epub']/year",
        ".//article-meta/pub-date[@pub-type='ppub']/year",
        ".//article-meta/pub-date/year",
    ):
        n = root.find(xp)
        if n is not None and n.text:
            return n.text.strip()
    return None


def _extract_authors(root: etree._Element) -> list[str]:
    out: list[str] = []
    for contrib in root.findall(".//contrib-group/contrib[@contrib-type='author']"):
        surname = _text_of(contrib.find(".//name/surname"))
        given = _text_of(contrib.find(".//name/given-names"))
        full = " ".join(p for p in (given, surname) if p).strip()
        if full:
            out.append(full)
    return out


def _extract_sections(body: etree._Element | None) -> list[tuple[str, str]]:
    if body is None:
        return []
    sections: list[tuple[str, str]] = []
    for sec in body.findall("./sec"):
        title = _text_of(sec.find("./title")) or "Section"
        paragraphs = [_text_of(p) for p in sec.findall(".//p")]
        text = "\n\n".join(p for p in paragraphs if p)
        if text:
            sections.append((title, text))
    if not sections:
        # Fall back to all top-level paragraphs if the body has no <sec>.
        paragraphs = [_text_of(p) for p in body.findall(".//p")]
        text = "\n\n".join(p for p in paragraphs if p)
        if text:
            sections.append(("Body", text))
    return sections


def parse_jats(xml: bytes | str | Path) -> ParsedArticle:
    if isinstance(xml, Path):
        xml = xml.read_bytes()
    if isinstance(xml, str):
        xml = xml.encode("utf-8")

    # PMC XML occasionally carries DTDs that slow or break strict parsers.
    parser = etree.XMLParser(recover=True, resolve_entities=False, load_dtd=False, no_network=True)
    root = etree.fromstring(xml, parser=parser)

    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        raise ValueError("No <article> element found in JATS XML")

    pmcid, pmid, doi = _extract_ids(article)
    abstract_node = article.find(".//article-meta/abstract")

    return ParsedArticle(
        pmcid=pmcid,
        pmid=pmid,
        doi=doi,
        title=_find_meta(article, ".//article-meta/title-group/article-title"),
        journal=_find_meta(article, ".//journal-meta/journal-title-group/journal-title")
        or _find_meta(article, ".//journal-meta/journal-title"),
        year=_extract_year(article),
        authors=_extract_authors(article),
        abstract=_text_of(abstract_node),
        sections=_extract_sections(article.find("./body")),
    )
