"""LLM-driven extraction of user-defined artefacts from parsed articles."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .backends import LLMBackend, build_backend
from .parser import ParsedArticle

FieldType = Literal["string", "number", "integer", "boolean", "list[string]", "object"]


@dataclass
class FieldSpec:
    """One artefact field the user wants extracted.

    `name` becomes the JSON key. `description` is the entire instruction the LLM
    sees for this field, so be specific about what counts as a valid value and
    what should return null.
    """
    name: str
    description: str
    type: FieldType = "string"
    required: bool = False
    enum: list[str] | None = None
    example: Any = None

    def to_json_schema(self) -> dict[str, Any]:
        base: dict[str, Any]
        if self.type == "list[string]":
            base = {"type": "array", "items": {"type": "string"}}
        elif self.type == "object":
            base = {"type": "object"}
        else:
            base = {"type": self.type}
        base["description"] = self.description
        if self.enum:
            base["enum"] = list(self.enum)
        if self.example is not None:
            base["examples"] = [self.example]
        return base


@dataclass
class ArtefactSchema:
    """User-defined collection of fields to extract.

    Use `mode="per_article"` to extract one record per article (default), or
    `mode="collection"` to have the LLM synthesize a single record across the
    whole set of articles (useful for meta-analyses, evidence summaries, etc.).
    """
    name: str
    fields: list[FieldSpec]
    instructions: str = ""
    mode: Literal["per_article", "collection"] = "per_article"
    allow_missing: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtefactSchema":
        fields = [FieldSpec(**f) for f in data.get("fields", [])]
        return cls(
            name=data.get("name", "artefact"),
            fields=fields,
            instructions=data.get("instructions", ""),
            mode=data.get("mode", "per_article"),
            allow_missing=data.get("allow_missing", True),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ArtefactSchema":
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("pyyaml not installed; use JSON or `pip install pyyaml`") from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_json_schema(self) -> dict[str, Any]:
        props = {f.name: f.to_json_schema() for f in self.fields}
        required = [f.name for f in self.fields if f.required]
        schema: dict[str, Any] = {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema


@dataclass
class ExtractionResult:
    source: str  # PMCID or "collection"
    data: dict[str, Any]
    raw: str = ""
    errors: list[str] = field(default_factory=list)


_SYSTEM_PROMPT = """You extract structured information from biomedical full-text articles.

Rules:
- Only use information present in the provided article text. Do not invent or infer beyond what is stated.
- If a field's value is not reported, return null (or an empty list for list fields), unless required=true and the value is clearly present.
- Return a single JSON object conforming exactly to the provided JSON schema. No prose, no markdown fences."""


class ArtefactExtractor:
    """Runs an LLM call per article (or once across a collection) using a user schema.

    The backend is pluggable: pass an `LLMBackend` (e.g. `AnthropicBackend` or
    `OllamaBackend`) explicitly, or use the convenience `backend=`/`model=` kwargs
    which delegate to `build_backend`.
    """

    def __init__(
        self,
        schema: ArtefactSchema,
        backend: LLMBackend | str = "anthropic",
        model: str | None = None,
        **backend_kwargs: Any,
    ):
        self.schema = schema
        if isinstance(backend, str):
            self.backend = build_backend(backend, model=model, **backend_kwargs)
        else:
            self.backend = backend

    def _build_user_prompt(self, articles: Iterable[ParsedArticle]) -> str:
        js = json.dumps(self.schema.to_json_schema(), indent=2)
        header = (
            f"Artefact: {self.schema.name}\n"
            f"JSON schema the response must conform to:\n```json\n{js}\n```\n"
        )
        if self.schema.instructions:
            header += f"\nAdditional instructions:\n{self.schema.instructions}\n"

        chunks: list[str] = []
        for i, art in enumerate(articles, 1):
            ref = art.pmcid or art.doi or art.pmid or f"article {i}"
            chunks.append(f"--- ARTICLE {i} [{ref}] ---\n{art.as_text()}")

        task = (
            "Extract the artefact and respond with a single JSON object matching the schema. "
            "Do not wrap in markdown, do not add commentary."
        )
        return f"{header}\n{task}\n\n" + "\n\n".join(chunks)

    def _call(self, prompt: str) -> str:
        return self.backend.complete(
            system=_SYSTEM_PROMPT,
            user=prompt,
            json_schema=self.schema.to_json_schema(),
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in model output: {raw[:200]}")
        return json.loads(s[start : end + 1])

    def extract_one(self, article: ParsedArticle) -> ExtractionResult:
        prompt = self._build_user_prompt([article])
        raw = self._call(prompt)
        source = article.pmcid or article.doi or article.pmid or "article"
        try:
            data = self._parse_json(raw)
            return ExtractionResult(source=source, data=data, raw=raw)
        except Exception as e:  # noqa: BLE001
            return ExtractionResult(source=source, data={}, raw=raw, errors=[str(e)])

    def extract_collection(self, articles: list[ParsedArticle]) -> ExtractionResult:
        prompt = self._build_user_prompt(articles)
        raw = self._call(prompt)
        try:
            data = self._parse_json(raw)
            return ExtractionResult(source="collection", data=data, raw=raw)
        except Exception as e:  # noqa: BLE001
            return ExtractionResult(source="collection", data={}, raw=raw, errors=[str(e)])

    def extract(self, articles: list[ParsedArticle]) -> list[ExtractionResult]:
        if self.schema.mode == "collection":
            return [self.extract_collection(articles)]
        return [self.extract_one(a) for a in articles]
