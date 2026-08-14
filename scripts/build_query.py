"""
Synthesize a PubMed query for a topic, from a plain-language description plus optional
example papers.

Examples
--------
Description only::

    python scripts/build_query.py --topic "chest x-ray datasets"

Description plus papers that must (and must not) be retrieved::

    python scripts/build_query.py \\
        --topic "chest x-ray datasets" \\
        --must-include 31831740 10.1148/ryai.210315 \\
        --must-exclude 12345678 \\
        --audit-sample 20

Refine an existing hand-written query instead of starting from scratch::

    python scripts/build_query.py --topic "radiology datasets" --start-modality radiology
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from Bio import Entrez

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radiology_dataset_db.config import LOG_LEVEL, MODEL, PUBMED_QUERY_DICT, SUPPORTED_MODALITIES
from radiology_dataset_db.query_synthesis import build_topic_query

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug or "topic"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize a PubMed query for a topic using an LLM plus live PubMed feedback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-t", "--topic", type=str, required=True, help="Plain-language description of what you want, e.g. 'chest x-ray datasets'.")
    parser.add_argument("--must-include", type=str, nargs="*", default=[], help="Papers that MUST be retrieved. Accepts PMIDs, DOIs, PubMed URLs, doi.org URLs, or PMCIDs.")
    parser.add_argument("--must-exclude", type=str, nargs="*", default=[], help="Papers that must NOT be retrieved. Same accepted formats.")
    parser.add_argument("--seeds-file", type=str, default=None, help="Text file of must-include identifiers, one per line ('#' comments allowed).")

    parser.add_argument("-o", "--output-path", type=str, default=None, help="Where to write the topic spec JSON. Defaults to topics/{slug}.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")

    parser.add_argument("-max", "--max-results", type=int, default=15000, help="Upper bound on PubMed hits; the agent prunes terms until the query fits.")
    parser.add_argument("-min", "--min-results", type=int, default=100, help="Lower bound on PubMed hits; below this the agent broadens the query.")
    parser.add_argument("--max-iterations", type=int, default=12, help="Maximum repair iterations.")
    parser.add_argument("--max-prune-steps", type=int, default=8, help="Maximum terms the agent may drop while shrinking the query.")

    parser.add_argument("--start-query", type=str, default=None, help="Start from this query instead of proposing one from scratch.")
    parser.add_argument("--start-modality", type=str, default=None, choices=sorted(SUPPORTED_MODALITIES), help="Start from the existing hand-written query for this modality.")

    parser.add_argument("--no-verify-seeds", action="store_true", help="Skip the LLM check that each must-include paper is on topic and introduces a dataset.")
    parser.add_argument("--keep-rejected-seeds", action="store_true", help="Anchor the query on must-include papers even if they fail verification.")
    parser.add_argument("--audit-sample", type=int, default=0, help="Sample this many results and have the LLM judge them, to estimate precision. 0 disables.")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"topics/{slugify(args.topic)}.json"

    if args.start_modality:
        if args.start_query:
            parser.error("Pass only one of --start-query and --start-modality.")
        start_query = " ".join((PUBMED_QUERY_DICT.get(args.start_modality) or "").split())
        if not start_query:
            parser.error(f"No query is defined for modality {args.start_modality!r} in config.PUBMED_QUERY_DICT.")
        args.start_query = start_query

    if args.seeds_file:
        seed_path = Path(args.seeds_file)
        if not seed_path.exists():
            parser.error(f"Seeds file not found: {seed_path}")
        extra = [
            line.strip()
            for line in seed_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        args.must_include = list(args.must_include) + extra

    if args.min_results > args.max_results:
        parser.error("--min-results cannot exceed --max-results.")

    return args


def getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set.")
    return value


def print_report(result) -> None:
    evaluation = result.evaluation

    print("\n" + "=" * 78)
    print(f"TOPIC: {result.topic}")
    print("=" * 78)
    print(f"\nQUERY ({evaluation.count} PubMed results):\n")
    print(result.query)

    print("\nSEEDS")
    for seed in result.seeds:
        if seed.rejected_reason:
            status = f"REJECTED ({seed.rejected_reason})"
        elif seed.polarity == "negative":
            status = "excluded" if seed.pmid not in evaluation.caught_negative else "STILL PRESENT"
        else:
            status = "found" if seed.pmid in evaluation.found_positive else "MISSING"
        print(f"  [{status:>22}] {seed.pmid}  {seed.title[:80]}")

    if result.unresolved:
        print("\nUNRESOLVED IDENTIFIERS")
        for identifier in result.unresolved:
            print(f"  {identifier}")

    recall = evaluation.seed_recall
    print("\nOUTCOME")
    print(f"  converged:    {result.converged}")
    print(f"  seed recall:  {'n/a' if recall is None else f'{recall:.0%}'}")
    print(f"  result count: {evaluation.count}")

    if result.term_contributions:
        print("\nTOP TERM CONTRIBUTIONS (hits lost if the term is dropped)")
        for block, term, hits in result.term_contributions[:10]:
            share = (hits / evaluation.count) if evaluation.count else 0
            print(f"  {hits:>7}  ({share:>5.1%})  [{block}] {term}")

    if result.audit:
        audit = result.audit
        precision = audit.get("estimated_precision")
        print("\nPRECISION AUDIT")
        print(f"  sampled: {audit.get('sampled')} of {audit.get('total_results')}")
        print(f"  on topic AND introduces a dataset: {audit.get('on_topic_and_dataset')}")
        if precision is not None:
            print(f"  estimated precision: {precision:.0%}")

    print()


async def main():
    args = parse_args()

    Entrez.email = getenv("ENTREZ_EMAIL")
    if os.getenv("ENTREZ_API_KEY"):
        Entrez.api_key = getenv("ENTREZ_API_KEY")
        logger.info("Using Entrez API key")

    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        logger.error(f"Output file {output_path} already exists. Pass --overwrite to replace it.")
        return 1

    logger.info(f"Using model: {MODEL}")
    logger.info(f"Synthesizing a PubMed query for topic: {args.topic!r}")

    result = await build_topic_query(
        topic=args.topic,
        must_include=args.must_include,
        must_exclude=args.must_exclude,
        verify_seeds=not args.no_verify_seeds,
        keep_rejected_seeds=args.keep_rejected_seeds,
        start_query=args.start_query,
        audit_sample=args.audit_sample,
        max_results=args.max_results,
        min_results=args.min_results,
        max_iterations=args.max_iterations,
        max_prune_steps=args.max_prune_steps,
    )

    if output_path.parent and str(output_path.parent) != "":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2))

    print_report(result)
    logger.info(f"Topic spec saved to {output_path}")
    logger.info(f"Run the extraction pipeline with: python scripts/build_db.py --topic-spec {output_path} --database-modality MODALITY")

    if result.evaluation.missing_positive:
        logger.warning("Some must-include papers are still missing from the results; review the query before using it.")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.warning("Interrupted; no topic spec was written.")
        raise SystemExit(130)
