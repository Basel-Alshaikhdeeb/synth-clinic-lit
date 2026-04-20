"""Programmatic end-to-end example.

Run from the repo root:
    python -m examples.example_usage
"""
from pathlib import Path

from synthclinclit import (
    ArtefactExtractor,
    ArtefactSchema,
    PMCDownloader,
    parse_jats,
    resolve_ids,
)

INPUT_IDS = [
    "10.1371/journal.pone.0173955",   # DOI
    "PMC5389857",                     # PMCID
    "28362821",                       # PMID
]

OUT_DIR = Path("./articles")


def main() -> None:
    resolved = resolve_ids(INPUT_IDS)
    for r in resolved:
        print(f"{r.input_id}: pmcid={r.pmcid} pmid={r.pmid} doi={r.doi} err={r.error}")

    articles = []
    with PMCDownloader() as dl:
        for r in resolved:
            if not r.pmcid:
                continue
            availability = dl.check_availability(r.pmcid)
            if not availability.oa_available:
                print(f"  {r.pmcid}: no OA full text ({availability.reason})")
                continue
            path = dl.download(r, OUT_DIR)
            if path:
                articles.append(parse_jats(path))

    if not articles:
        print("No articles retrieved.")
        return

    schema = ArtefactSchema.from_file("examples/schema_clinical_trial.json")
    extractor = ArtefactExtractor(schema=schema)
    for result in extractor.extract(articles):
        print(f"\n--- {result.source} ---")
        print(result.data)


if __name__ == "__main__":
    main()
