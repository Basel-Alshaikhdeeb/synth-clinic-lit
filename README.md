# SynthClincLit

Download full-text biomedical articles from PubMed Central by DOI, PMID, or PMCID, then use a large language model to extract user-defined artefacts from a single article or across a collection.

## What it does

1. **Resolve** — accepts a mix of DOIs, PMIDs, and PMCIDs and resolves them to the full triple via the NCBI ID Converter.
2. **Check availability** — queries the PMC Open Access service before downloading so you know whether full text is retrievable.
3. **Download** — fetches JATS XML full text via NCBI EFetch and writes one file per article.
4. **Parse** — extracts title, authors, abstract, and body sections from JATS XML.
5. **Extract** — sends the parsed article(s) to a Claude model with a JSON schema you defined, returning structured artefacts either **per article** or **across the whole collection**.

## Install

```bash
pip install -e .
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and NCBI_EMAIL
```

NCBI E-utilities require a tool name and email. An `NCBI_API_KEY` is optional but raises your rate limit from 3 to 10 requests/second.

## Define the artefact you want

The artefact to extract is defined in a JSON (or YAML) schema file. Each field has a name, type, and the description that the model will read. See `examples/schema_clinical_trial.json`.

```json
{
  "name": "clinical_trial_summary",
  "mode": "per_article",
  "fields": [
    {"name": "n_participants", "type": "integer",
     "description": "Total number of human participants enrolled. Null if not reported."},
    {"name": "primary_outcomes", "type": "list[string]",
     "description": "Primary outcome measures as defined by the authors."}
  ]
}
```

Supported field types: `string`, `number`, `integer`, `boolean`, `list[string]`, `object`. Optional `enum`, `required`, and `example` per field.

- `mode: "per_article"` — one extraction per article (the default).
- `mode: "collection"` — one synthesized extraction across all articles. Useful for evidence summaries / meta-analysis prep. See `examples/schema_collection_meta.json`.

## Free local extraction via Ollama

Run extractions on your own machine with open-weights models — zero API cost.

**1. Install Ollama** — https://ollama.com (macOS/Linux/Windows).

**2. Pull a model.** For JATS-length biomedical articles a ~8–12 B model with a large context window is a good tradeoff between quality and VRAM:

| Model | Ollama tag | Size | Default ctx | Good at |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | `llama3.1:8b` | ~4.7 GB | 128K | General, reliable JSON |
| Gemma 2 9B Instruct | `gemma2:9b` | ~5.4 GB | 8K | Strong summarization |
| Gemma 3 12B Instruct | `gemma3:12b` | ~8.1 GB | 128K | Better reasoning, longer ctx |
| Qwen 2.5 7B Instruct | `qwen2.5:7b` | ~4.7 GB | 128K | Very strong JSON adherence |
| Mistral Nemo 12B Instruct | `mistral-nemo:12b` | ~7.1 GB | 128K | Good multilingual, 128K ctx |
| Phi-4 14B | `phi4:14b` | ~9.1 GB | 16K | Strong reasoning |
| Biomistral 7B | `cniongolo/biomistral` | ~4.4 GB | 4K | Domain-tuned on PubMed |

RAM/VRAM rough guide: Q4 quantization (Ollama's default) needs roughly `size_gb + 2 GB` free. An 8B model runs comfortably on 8 GB VRAM or a 16 GB Apple-silicon Mac.

```bash
ollama pull llama3.1:8b
ollama serve    # usually auto-started; leaves REST API at localhost:11434
```

**3. Run extraction against Ollama** — no API key needed:

```bash
synthclinclit extract \
    -s examples/schema_clinical_trial.json \
    -f ids.txt \
    --backend ollama \
    --model llama3.1:8b \
    --num-ctx 16384        # bump ctx for long full-text articles
```

Programmatic equivalent:

```python
from synthclinclit import ArtefactExtractor, ArtefactSchema, OllamaBackend

backend = OllamaBackend(model="qwen2.5:7b", num_ctx=16384)
extractor = ArtefactExtractor(ArtefactSchema.from_file("schema.json"), backend=backend)
```

**Reliability tip.** Ollama supports passing a JSON schema in the `format` field to constrain decoding — this package does that automatically, which dramatically improves schema conformance for small models. If a field keeps coming back wrong, tighten its `description` in the schema or add an `enum`/`example`.

**Context window.** Full-text articles frequently exceed the default 8 K context. Use a model that supports 32 K+ (Llama 3.1, Qwen 2.5, Gemma 3, Mistral Nemo all do) and pass `--num-ctx 16384` or higher. Gemma 2 caps at 8 K; pre-trim sections before extraction if you use it.

## CLI

```bash
# Resolve a mixed list
synthclinclit resolve -i 10.1371/journal.pone.0173955 -i PMC5389857 -i 28362821

# Check OA availability (one per line in ids.txt)
synthclinclit check -f ids.txt

# Download JATS XML to ./articles/
synthclinclit download -f ids.txt -o ./articles --require-oa

# Extract artefacts with your schema (default: Anthropic / Opus 4.7)
synthclinclit extract -s examples/schema_clinical_trial.json -f ids.txt -r extractions.json

# Or free, local extraction via Ollama
synthclinclit extract -s examples/schema_clinical_trial.json -f ids.txt \
    --backend ollama --model llama3.1:8b --num-ctx 16384

# Or extract from already-downloaded XML
synthclinclit extract -s examples/schema_clinical_trial.json --xml-dir ./articles
```

## Programmatic use

```python
from synthclinclit import resolve_ids, PMCDownloader, parse_jats, ArtefactExtractor, ArtefactSchema

resolved = resolve_ids(["10.1371/journal.pone.0173955", "PMC5389857"])
articles = []
with PMCDownloader() as dl:
    for r in resolved:
        if r.pmcid and dl.check_availability(r.pmcid).oa_available:
            path = dl.download(r, out_dir="./articles")
            if path:
                articles.append(parse_jats(path))

schema = ArtefactSchema.from_file("examples/schema_clinical_trial.json")
results = ArtefactExtractor(schema).extract(articles)
```

## Notes & limits

- Non-OA articles: EFetch can still return XML for many non-OA PMC records, but some will be blocked. Pass `--require-oa` to the CLI or check `availability.oa_available` to skip those up front.
- The ID Converter only knows about articles indexed in PubMed/PMC. Preprints and non-indexed DOIs will come back with `error: invalid article id`.
- The extractor sends the full article text to the model. For very long articles you may want to pre-filter sections before calling `ArtefactExtractor.extract`.
