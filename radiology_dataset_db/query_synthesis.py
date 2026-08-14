"""
Agentic PubMed query synthesis.

Given a plain-language topic ("chest x-ray datasets") and, optionally, a handful of
papers that must (or must not) appear in the results, this module searches for a PubMed
boolean query that:

  1. retrieves every verified must-include paper,
  2. retrieves none of the must-exclude papers,
  3. returns a result count inside a sane band (broad enough to generalize beyond the
     seeds, narrow enough not to drown the downstream extraction pipeline).

The LLM proposes vocabulary; PubMed decides. Every term an agent proposes is validated
(MeSH descriptors are checked against the real MeSH vocabulary) and every edit is kept
only if the measured hit counts and seed membership actually improve. That split is what
makes this an agent rather than a prompt chain: the number of iterations, and which
repair to attempt, are decided by measurements taken during the run.
"""

import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv

from radiology_dataset_db.config import LOG_LEVEL, MODEL
from radiology_dataset_db.is_database_paper_classifier_llm import llm_thinks_not_dataset_paper
from radiology_dataset_db.pubmed_utils import (
    extract_pubmed_metadata, fetch_pubmed_details, get_count, mesh_term_exists,
    pmids_matching_query, resolve_identifier_to_pmid, search_pubmed)
from radiology_dataset_db.query_spec import (
    BLOCK_NAMES, QuerySpec, is_mesh_term, make_term, normalize_term, split_term)
from radiology_dataset_db.query_synthesis_llm import (
    paper_is_on_topic, propose_broadening_terms, propose_exclusion_terms,
    propose_query_terms, propose_recovery_terms, verify_seed_paper)

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

# NCBI allows 3 requests/second without an API key and 10/second with one.
ENTREZ_SLEEP = 0.11 if os.getenv("ENTREZ_API_KEY") else 0.34

# Field tags that are safe to relax to when a [ti] term is too strict.
_TITLE_FIELD = "ti"
_TITLE_ABSTRACT_FIELD = "tiab"


# -----------------------------
# COUNT CACHE
# -----------------------------
_COUNT_CACHE: Dict[str, int] = {}


def cached_count(query: str) -> int:
    """PubMed hit count for a query, memoized (the hillclimb revisits many queries)."""
    key = " ".join((query or "").split())
    if not key:
        return 0
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]
    count = get_count(key)
    time.sleep(ENTREZ_SLEEP)
    _COUNT_CACHE[key] = count
    return count


def clear_count_cache() -> None:
    _COUNT_CACHE.clear()


# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass
class SeedPaper:
    """A user-supplied paper that anchors the search."""
    identifier: str
    polarity: str  # "positive" (must be retrieved) or "negative" (must not be)
    pmid: Optional[str] = None
    title: str = ""
    abstract: str = ""
    mesh_terms: List[str] = field(default_factory=list)
    year: Optional[int] = None
    link: Optional[str] = None
    is_on_topic: Optional[bool] = None
    presents_dataset: Optional[bool] = None
    verification_reason: str = ""
    rejected_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.pmid is not None and not self.rejected_reason

    def summary(self) -> str:
        mesh = ", ".join(self.mesh_terms) if self.mesh_terms else "(none indexed)"
        abstract = (self.abstract or "")[:700]
        return (
            f"- PMID {self.pmid}: {self.title}\n"
            f"  MeSH: {mesh}\n"
            f"  Abstract: {abstract}"
        )


@dataclass
class QueryEvaluation:
    query: str
    count: int
    found_positive: List[str] = field(default_factory=list)
    missing_positive: List[str] = field(default_factory=list)
    caught_negative: List[str] = field(default_factory=list)

    @property
    def num_positive(self) -> int:
        return len(self.found_positive) + len(self.missing_positive)

    @property
    def seed_recall(self) -> Optional[float]:
        if self.num_positive == 0:
            return None
        return len(self.found_positive) / self.num_positive

    @property
    def seeds_satisfied(self) -> bool:
        return not self.missing_positive and not self.caught_negative


@dataclass
class IterationRecord:
    iteration: int
    action: str
    detail: str
    query: str
    count: int
    seed_recall: Optional[float]
    missing_positive: List[str] = field(default_factory=list)
    caught_negative: List[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    topic: str
    query: str
    spec: QuerySpec
    evaluation: QueryEvaluation
    seeds: List[SeedPaper]
    history: List[IterationRecord]
    converged: bool
    unresolved: List[str] = field(default_factory=list)
    term_contributions: List[Tuple[str, str, int]] = field(default_factory=list)
    audit: Optional[dict] = None
    model: str = MODEL

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "model": self.model,
            "query": self.query,
            "spec": self.spec.to_dict(),
            "result_count": self.evaluation.count,
            "converged": self.converged,
            "seed_recall": self.evaluation.seed_recall,
            "missing_positive_seeds": self.evaluation.missing_positive,
            "caught_negative_seeds": self.evaluation.caught_negative,
            "unresolved_identifiers": self.unresolved,
            "seeds": [asdict(seed) for seed in self.seeds],
            "term_contributions": [
                {"block": block, "term": term, "hits_contributed": hits}
                for block, term, hits in self.term_contributions
            ],
            "history": [asdict(record) for record in self.history],
            "audit": self.audit,
        }


# -----------------------------
# SEED RESOLUTION AND VERIFICATION
# -----------------------------
def resolve_seed_papers(identifiers: Sequence[str], polarity: str) -> Tuple[List[SeedPaper], List[str]]:
    """Turn user-supplied PMIDs/DOIs/URLs into SeedPapers with PubMed metadata."""
    seeds: List[SeedPaper] = []
    unresolved: List[str] = []

    pending: List[SeedPaper] = []
    for identifier in identifiers:
        identifier = (identifier or "").strip()
        if not identifier:
            continue
        pmid = resolve_identifier_to_pmid(identifier)
        if not pmid:
            logger.warning(f"Could not resolve {identifier!r} to a PMID; skipping.")
            unresolved.append(identifier)
            continue
        pending.append(SeedPaper(identifier=identifier, polarity=polarity, pmid=pmid))

    if not pending:
        return seeds, unresolved

    articles = fetch_pubmed_details([seed.pmid for seed in pending])
    metadata_by_pmid = {}
    for article in articles:
        metadata = extract_pubmed_metadata(article)
        if metadata.get("pmid"):
            metadata_by_pmid[str(metadata["pmid"])] = metadata

    for seed in pending:
        metadata = metadata_by_pmid.get(seed.pmid)
        if not metadata:
            logger.warning(f"No PubMed record fetched for PMID {seed.pmid} ({seed.identifier}); skipping.")
            unresolved.append(seed.identifier)
            continue
        seed.title = metadata.get("title") or ""
        seed.abstract = metadata.get("abstract") or ""
        seed.mesh_terms = metadata.get("mesh_terms") or []
        seed.year = metadata.get("year")
        seed.link = metadata.get("link")
        seeds.append(seed)

    return seeds, unresolved


async def verify_seed_papers(topic: str, seeds: Sequence[SeedPaper], require_dataset: bool = True) -> None:
    """
    Ask the LLM whether each positive seed really is an on-topic dataset paper.

    Seeds that fail are marked with a rejected_reason and dropped from the anchor set, so
    that a mistaken example cannot drag the whole query off topic. Negative seeds are
    checked too, but only to warn the user when a must-exclude paper looks on topic.
    """
    for seed in seeds:
        verification = await verify_seed_paper(topic, seed.title, seed.abstract)
        if verification is None:
            logger.warning(f"Could not verify PMID {seed.pmid}; keeping it without verification.")
            continue

        seed.is_on_topic = verification.is_on_topic
        seed.presents_dataset = verification.presents_dataset
        seed.verification_reason = verification.reason or ""

        if seed.polarity == "positive":
            if not verification.is_on_topic:
                seed.rejected_reason = "not on topic"
            elif require_dataset and not verification.presents_dataset:
                seed.rejected_reason = "does not introduce a dataset"
            if seed.rejected_reason:
                logger.warning(
                    f"Rejecting seed PMID {seed.pmid} ({seed.title[:70]}...): "
                    f"{seed.rejected_reason}. Reason given: {seed.verification_reason}"
                )
        elif verification.is_on_topic and verification.presents_dataset:
            logger.warning(
                f"Must-exclude PMID {seed.pmid} looks like an on-topic dataset paper "
                f"({seed.verification_reason}). Excluding it may cost real recall."
            )


def format_seed_summaries(seeds: Sequence[SeedPaper]) -> str:
    if not seeds:
        return "(none provided -- rely on your own knowledge of the field)"
    return "\n".join(seed.summary() for seed in seeds)


# -----------------------------
# TERM GROUNDING
# -----------------------------
def normalize_proposed_term(term: str) -> Optional[str]:
    """
    Clean up one LLM-proposed term.

    Rejects terms containing boolean operators or parentheses (the agent is asked for one
    concept per term), and defaults an untagged term to [tiab].
    """
    term = normalize_term(term)
    if not term:
        return None
    if any(ch in term for ch in "()"):
        logger.debug(f"Discarding proposed term with parentheses: {term!r}")
        return None
    if any(f" {op} " in f" {term.upper()} " for op in ("AND", "OR", "NOT")):
        logger.debug(f"Discarding proposed term containing a boolean operator: {term!r}")
        return None

    text, field_tag = split_term(term)
    if not text:
        return None
    if field_tag is None:
        return make_term(text, _TITLE_ABSTRACT_FIELD)
    return make_term(text, field_tag)


def ground_terms(terms: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Validate proposed terms against reality.

    A [MeSH] term that is not a real MeSH descriptor is downgraded to [tiab] rather than
    dropped: the LLM's intent is usually right even when its vocabulary is not, and a
    hallucinated MeSH descriptor silently matches nothing at all in PubMed.

    Returns (grounded_terms, notes).
    """
    grounded: List[str] = []
    notes: List[str] = []

    for term in terms:
        normalized = normalize_proposed_term(term)
        if not normalized:
            notes.append(f"discarded malformed term: {term!r}")
            continue

        if is_mesh_term(normalized):
            text, _ = split_term(normalized)
            if not mesh_term_exists(text):
                downgraded = make_term(text, _TITLE_ABSTRACT_FIELD)
                notes.append(f"{normalized} is not a real MeSH descriptor -> {downgraded}")
                normalized = downgraded

        grounded.append(normalized)

    return grounded, notes


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_spec(
    spec: QuerySpec,
    positive_pmids: Sequence[str],
    negative_pmids: Sequence[str],
) -> QueryEvaluation:
    """Measure a query: total hits, which positive seeds it finds, which negatives it catches."""
    query = spec.render()
    if not query:
        return QueryEvaluation(query="", count=0, missing_positive=list(positive_pmids))

    count = cached_count(query)
    all_seeds = list(positive_pmids) + list(negative_pmids)
    matched = pmids_matching_query(all_seeds, query, sleep=ENTREZ_SLEEP) if all_seeds else set()

    return QueryEvaluation(
        query=query,
        count=count,
        found_positive=[p for p in positive_pmids if p in matched],
        missing_positive=[p for p in positive_pmids if p not in matched],
        caught_negative=[p for p in negative_pmids if p in matched],
    )


def diagnose_missing_seeds(spec: QuerySpec, missing_pmids: Sequence[str]) -> Dict[str, List[str]]:
    """
    Work out *why* each missing seed is missing: which block(s) it fails.

    Costs one esearch per block rather than per paper. An empty block imposes no
    constraint, so every paper "matches" it.
    """
    missing_pmids = [str(p) for p in missing_pmids]
    if not missing_pmids:
        return {}

    matches_by_block: Dict[str, Set[str]] = {}
    for name in BLOCK_NAMES:
        block_query = spec.block_query(name)
        if not block_query:
            matches_by_block[name] = set(missing_pmids) if name != "exclude" else set()
            continue
        matches_by_block[name] = pmids_matching_query(missing_pmids, block_query, sleep=ENTREZ_SLEEP)

    diagnosis: Dict[str, List[str]] = {}
    for pmid in missing_pmids:
        failing = [name for name in ("dataset", "topic") if pmid not in matches_by_block[name]]
        if pmid in matches_by_block["exclude"]:
            failing.append("exclude")
        diagnosis[pmid] = failing

    return diagnosis


def term_contributions(spec: QuerySpec) -> List[Tuple[str, str, int]]:
    """
    Drop-one analysis of the final query: how many hits each term is responsible for.

    This is the evidence for whether a query "pulls excessively" -- a term contributing a
    large share of the total is exactly the situation that motivated dropping
    "Databases, Factual"[MeSH] from the hand-written radiology query.
    """
    base = cached_count(spec.render())
    contributions: List[Tuple[str, str, int]] = []

    for name, term in list(spec.iter_terms()):
        if len(spec.block(name)) <= 1:
            continue
        reduced = spec.with_term_removed(name, term)
        if not reduced.is_runnable():
            continue
        contributions.append((name, term, base - cached_count(reduced.render())))

    contributions.sort(key=lambda row: row[2], reverse=True)
    return contributions


# -----------------------------
# REPAIR STEPS
# -----------------------------
def _seed_by_pmid(seeds: Sequence[SeedPaper], pmid: str) -> Optional[SeedPaper]:
    for seed in seeds:
        if seed.pmid == pmid:
            return seed
    return None


def _candidate_recovers(spec: QuerySpec, block_name: str, terms: Sequence[str], pmid: str) -> bool:
    """Does adding these terms to this block make the target paper match the block?"""
    candidate = spec.with_terms_added(block_name, terms)
    block_query = candidate.block_query(block_name)
    if not block_query:
        return False
    return pmid in pmids_matching_query([pmid], block_query, sleep=ENTREZ_SLEEP)


def _deterministic_recovery_candidates(seed: SeedPaper, spec: QuerySpec, block_name: str) -> List[str]:
    """
    Fallback vocabulary when the LLM's proposal fails to recover a seed.

    - topic block: the paper's own PubMed-indexed MeSH terms, which are guaranteed to match.
    - dataset block: [tiab] relaxations of the existing [ti] terms, which is the natural
      repair for a dataset paper that does not announce itself in its title.
    """
    if block_name == "topic":
        return [make_term(term, "MeSH") for term in seed.mesh_terms]

    if block_name == "dataset":
        relaxed = []
        for term in spec.block("dataset"):
            text, field_tag = split_term(term)
            if field_tag and field_tag.lower() == _TITLE_FIELD:
                relaxed.append(make_term(text, _TITLE_ABSTRACT_FIELD))
        return relaxed

    return []


def _cheapest_recovering_term(
    spec: QuerySpec,
    block_name: str,
    candidates: Sequence[str],
    pmid: str,
    max_added_hits: Optional[int] = None,
) -> Optional[str]:
    """
    Of the candidate terms that individually recover the target paper, pick the one that
    adds the fewest new hits -- generality is preferred, but not at any price.
    """
    base = cached_count(spec.render())
    best: Optional[Tuple[int, str]] = None

    for term in candidates:
        if not _candidate_recovers(spec, block_name, [term], pmid):
            continue
        added = cached_count(spec.with_terms_added(block_name, [term]).render()) - base
        if max_added_hits is not None and added > max_added_hits:
            logger.debug(f"Skipping recovery term {term!r}: adds {added} hits (cap {max_added_hits}).")
            continue
        if best is None or added < best[0]:
            best = (added, term)

    return best[1] if best else None


def _drop_exclusions_catching(spec: QuerySpec, pmid: str) -> Tuple[QuerySpec, List[str]]:
    """Remove NOT-block terms that are wrongly filtering out a required paper."""
    dropped: List[str] = []
    for term in list(spec.block("exclude")):
        if pmid in pmids_matching_query([pmid], f"({term})", sleep=ENTREZ_SLEEP):
            spec = spec.with_term_removed("exclude", term)
            dropped.append(term)
    return spec, dropped


def prune_step(
    spec: QuerySpec,
    positive_pmids: Sequence[str],
    max_results: int,
    min_results: int,
) -> Optional[Tuple[QuerySpec, str, int]]:
    """
    One greedy prune: remove the single term that costs the most hits while costing no
    required paper. Terms contributing nothing at all are removed first, regardless of
    whether the query is over budget.

    The papers protected from pruning are the must-include papers the query *currently*
    retrieves, not every must-include paper. Guarding on the full list would let one
    unreachable seed veto every prune for the rest of the run.

    Returns (new_spec, term_label, hits_removed), or None if no safe prune exists.
    """
    base = cached_count(spec.render())
    protected: Set[str] = set()
    if positive_pmids:
        protected = pmids_matching_query(positive_pmids, spec.render(), sleep=ENTREZ_SLEEP)

    def keeps_protected(candidate_spec: QuerySpec) -> bool:
        if not protected:
            return True
        retained = pmids_matching_query(protected, candidate_spec.render(), sleep=ENTREZ_SLEEP)
        return len(retained) == len(protected)

    candidates: List[Tuple[int, str, str]] = []  # (hits_removed, block, term)

    for name in ("dataset", "topic"):
        terms = spec.block(name)
        if len(terms) <= 1:
            continue  # never empty a block: that would change the query's shape
        for term in list(terms):
            reduced = spec.with_term_removed(name, term)
            candidates.append((base - cached_count(reduced.render()), name, term))

    if not candidates:
        return None

    for _, name, term in [c for c in candidates if c[0] <= 0]:
        reduced = spec.with_term_removed(name, term)
        if keeps_protected(reduced):
            return reduced, f"{name}:{term}", 0

    if base <= max_results:
        return None

    for hits_removed, name, term in sorted(candidates, key=lambda row: row[0], reverse=True):
        if hits_removed <= 0:
            continue
        reduced = spec.with_term_removed(name, term)
        reduced_count = base - hits_removed
        if reduced_count < min_results:
            logger.debug(f"Not dropping {term!r}: would leave only {reduced_count} results (floor {min_results}).")
            continue
        if not keeps_protected(reduced):
            logger.debug(f"Not dropping {term!r}: it is required by a must-include paper.")
            continue
        return reduced, f"{name}:{term}", hits_removed

    return None


# -----------------------------
# MAIN LOOP
# -----------------------------
async def synthesize_query(
    topic: str,
    positive_seeds: Sequence[SeedPaper] = (),
    negative_seeds: Sequence[SeedPaper] = (),
    start_spec: Optional[QuerySpec] = None,
    max_results: int = 15000,
    min_results: int = 100,
    max_iterations: int = 12,
    max_prune_steps: int = 8,
    max_recovery_added_hits: Optional[int] = None,
    unresolved: Sequence[str] = (),
) -> SynthesisResult:
    """
    Hillclimb to a query that satisfies the seeds and lands inside the result-count band.

    Each iteration measures the current query and applies exactly one repair, in priority
    order: recover missing must-include papers, then exclude must-exclude papers, then
    prune for size, then broaden if implausibly narrow.
    """
    positive_pmids = [s.pmid for s in positive_seeds if s.pmid]
    negative_pmids = [s.pmid for s in negative_seeds if s.pmid]
    all_seeds = list(positive_seeds) + list(negative_seeds)
    history: List[IterationRecord] = []

    if max_recovery_added_hits is None:
        max_recovery_added_hits = max(max_results, 1) * 2

    # -- initial proposal --------------------------------------------------
    if start_spec is not None:
        spec = start_spec.normalized()
        logger.info(f"Starting from supplied query: {spec.render()}")
    else:
        proposal = await propose_query_terms(topic, format_seed_summaries(positive_seeds))
        if proposal is None:
            raise RuntimeError(
                "The query proposal agent returned nothing. Check that the LLM backend "
                "(see MODEL/VLLM_PORT in your .env) is reachable."
            )
        spec = QuerySpec()
        for name, proposed in (
            ("dataset", proposal.dataset_terms),
            ("topic", proposal.topic_terms),
            ("exclude", proposal.exclude_terms),
        ):
            grounded, notes = ground_terms(proposed)
            for note in notes:
                logger.info(f"[{name}] {note}")
            spec.set_block(name, grounded)
        logger.info(f"Initial query: {spec.render()}")

    if not spec.is_runnable():
        raise RuntimeError("The proposed query has no dataset or topic terms; cannot search PubMed.")

    evaluation = evaluate_spec(spec, positive_pmids, negative_pmids)
    history.append(IterationRecord(
        iteration=0, action="propose", detail="initial query",
        query=evaluation.query, count=evaluation.count, seed_recall=evaluation.seed_recall,
        missing_positive=list(evaluation.missing_positive),
        caught_negative=list(evaluation.caught_negative),
    ))
    logger.info(
        f"Iteration 0: {evaluation.count} results, "
        f"{len(evaluation.found_positive)}/{evaluation.num_positive} must-include papers found."
    )

    fixed_point = False
    prune_steps_used = 0
    broadening_attempts = 0
    unrecoverable: Set[str] = set()

    for iteration in range(1, max_iterations + 1):
        action: Optional[str] = None
        detail = ""

        # -- 1. recover missing must-include papers ------------------------
        missing_to_fix = [p for p in evaluation.missing_positive if p not in unrecoverable]
        if missing_to_fix:
            diagnosis = diagnose_missing_seeds(spec, missing_to_fix)
            repaired = False

            for pmid, failing_blocks in diagnosis.items():
                seed = _seed_by_pmid(all_seeds, pmid)
                if seed is None:
                    continue

                if "exclude" in failing_blocks:
                    spec, dropped = _drop_exclusions_catching(spec, pmid)
                    if dropped:
                        action = "recover"
                        detail = f"dropped exclusion term(s) {dropped} that filtered out PMID {pmid}"
                        repaired = True
                        break

                for block_name in [b for b in failing_blocks if b != "exclude"]:
                    proposal = await propose_recovery_terms(
                        topic=topic,
                        query=spec.render(),
                        block_name=block_name,
                        block_terms=spec.block(block_name),
                        title=seed.title,
                        abstract=seed.abstract,
                        mesh_terms=seed.mesh_terms,
                    )
                    candidates: List[str] = []
                    if proposal and proposal.terms:
                        grounded, notes = ground_terms(proposal.terms)
                        for note in notes:
                            logger.info(f"[recover:{block_name}] {note}")
                        candidates.extend(grounded)

                    chosen = _cheapest_recovering_term(
                        spec, block_name, candidates, pmid, max_added_hits=max_recovery_added_hits
                    )

                    if chosen is None:
                        fallback, notes = ground_terms(
                            _deterministic_recovery_candidates(seed, spec, block_name)
                        )
                        for note in notes:
                            logger.debug(f"[recover-fallback:{block_name}] {note}")
                        chosen = _cheapest_recovering_term(
                            spec, block_name, fallback, pmid, max_added_hits=max_recovery_added_hits
                        )
                        if chosen is not None:
                            logger.info(f"LLM proposal did not recover PMID {pmid}; used fallback term {chosen!r}.")

                    if chosen is not None:
                        spec = spec.with_terms_added(block_name, [chosen])
                        action = "recover"
                        detail = f"added {chosen!r} to '{block_name}' block to recover PMID {pmid}"
                        repaired = True
                        break

                if repaired:
                    break

            if not repaired:
                # Every missing paper was tried and none could be recovered, so stop
                # spending agent calls on them and move on to the other repairs.
                unrecoverable.update(missing_to_fix)
                logger.warning(
                    f"Could not recover must-include PMID(s) {missing_to_fix} without an "
                    "over-broad term. Continuing with the remaining repairs."
                )

        # -- 2. exclude must-exclude papers --------------------------------
        if action is None and evaluation.caught_negative:
            pmid = evaluation.caught_negative[0]
            seed = _seed_by_pmid(all_seeds, pmid)
            if seed is not None:
                proposal = await propose_exclusion_terms(
                    topic=topic,
                    query=spec.render(),
                    title=seed.title,
                    abstract=seed.abstract,
                    mesh_terms=seed.mesh_terms,
                    positive_summaries=format_seed_summaries(
                        [s for s in positive_seeds if s.accepted]
                    ),
                )
                grounded, notes = ground_terms(proposal.terms if proposal else [])
                for note in notes:
                    logger.info(f"[exclude] {note}")

                for term in grounded:
                    candidate = spec.with_terms_added("exclude", [term])
                    # An exclusion term is only acceptable if it removes the unwanted
                    # paper and keeps every required one.
                    still_caught = pmids_matching_query([pmid], candidate.render(), sleep=ENTREZ_SLEEP)
                    if still_caught:
                        continue
                    # Guard the must-include papers currently retrieved, so an already
                    # unreachable seed cannot veto every possible exclusion term.
                    if evaluation.found_positive:
                        retained = pmids_matching_query(
                            evaluation.found_positive, candidate.render(), sleep=ENTREZ_SLEEP
                        )
                        if len(retained) != len(evaluation.found_positive):
                            logger.info(f"Rejecting exclusion term {term!r}: it also removes a must-include paper.")
                            continue
                    spec = candidate
                    action = "exclude"
                    detail = f"added {term!r} to exclusion block to drop PMID {pmid}"
                    break

                if action is None:
                    logger.warning(
                        f"No safe exclusion term found for PMID {pmid}; it will remain in the results."
                    )
                    negative_pmids = [p for p in negative_pmids if p != pmid]

        # -- 3. prune for size ---------------------------------------------
        if action is None and prune_steps_used < max_prune_steps:
            pruned = prune_step(spec, positive_pmids, max_results, min_results)
            if pruned is not None:
                spec, term_label, hits_removed = pruned
                prune_steps_used += 1
                action = "prune"
                detail = f"dropped {term_label!r} ({hits_removed} hits removed)"

        # -- 4. broaden if implausibly narrow ------------------------------
        if action is None and evaluation.count < min_results and broadening_attempts < 2:
            broadening_attempts += 1
            proposal = await propose_broadening_terms(
                topic=topic,
                query=spec.render(),
                topic_terms=spec.block("topic"),
                current_count=evaluation.count,
                target_count=min_results,
            )
            grounded, notes = ground_terms(proposal.terms if proposal else [])
            for note in notes:
                logger.info(f"[broaden] {note}")
            new_terms = [t for t in grounded if t.casefold() not in {x.casefold() for x in spec.block("topic")}]
            if new_terms:
                candidate = spec.with_terms_added("topic", new_terms)
                if negative_pmids:
                    caught = pmids_matching_query(negative_pmids, candidate.render(), sleep=ENTREZ_SLEEP)
                    for term in list(new_terms):
                        if not caught:
                            break
                        trial = spec.with_terms_added("topic", [t for t in new_terms if t != term])
                        if not pmids_matching_query(negative_pmids, trial.render(), sleep=ENTREZ_SLEEP):
                            new_terms = [t for t in new_terms if t != term]
                            candidate = trial
                            caught = set()
                if new_terms:
                    spec = candidate
                    action = "broaden"
                    detail = f"added {len(new_terms)} topic term(s): {new_terms}"

        if action is None:
            fixed_point = True
            logger.info(f"Reached a fixed point after {iteration - 1} repair iteration(s).")
            break

        evaluation = evaluate_spec(spec, positive_pmids, negative_pmids)
        history.append(IterationRecord(
            iteration=iteration, action=action, detail=detail,
            query=evaluation.query, count=evaluation.count, seed_recall=evaluation.seed_recall,
            missing_positive=list(evaluation.missing_positive),
            caught_negative=list(evaluation.caught_negative),
        ))
        logger.info(
            f"Iteration {iteration} [{action}]: {detail} -> {evaluation.count} results, "
            f"{len(evaluation.found_positive)}/{evaluation.num_positive} must-include papers found."
        )

    # "Converged" means a fixed point AND every seed constraint satisfied. A fixed point
    # with a missing must-include paper is a stall, not a success.
    converged = fixed_point and evaluation.seeds_satisfied
    if not fixed_point:
        logger.warning(f"Stopped after the {max_iterations}-iteration budget without settling.")
    elif not converged:
        logger.warning(
            "Settled on a query that does not satisfy every seed constraint "
            f"(missing: {evaluation.missing_positive}, still present: {evaluation.caught_negative})."
        )

    return SynthesisResult(
        topic=topic,
        query=spec.render(),
        spec=spec,
        evaluation=evaluation,
        seeds=all_seeds,
        history=history,
        converged=converged,
        unresolved=list(unresolved),
        term_contributions=term_contributions(spec),
    )


# -----------------------------
# PRECISION AUDIT
# -----------------------------
async def audit_query_precision(
    topic: str,
    query: str,
    sample_size: int = 20,
    random_seed: int = 0,
    max_ids: int = 20000,
) -> dict:
    """
    Estimate precision by sampling the result set and asking the LLM to judge each paper.

    Hit count alone says whether a query is big; this says whether it is *right*. The
    sample is drawn with a fixed seed so the number in the topic spec is reproducible.
    """
    ids = search_pubmed(query, max_results=max_ids)
    if not ids:
        return {"sampled": 0, "note": "query returned no results"}

    sample = random.Random(random_seed).sample(ids, min(sample_size, len(ids)))
    articles = fetch_pubmed_details(sample)

    rows = []
    on_topic_count = 0
    dataset_count = 0
    both_count = 0

    for article in articles:
        metadata = extract_pubmed_metadata(article)
        title = metadata.get("title") or ""
        abstract = metadata.get("abstract") or ""

        on_topic = await paper_is_on_topic(topic, title, abstract)
        is_dataset = not await llm_thinks_not_dataset_paper(title, abstract)

        if on_topic:
            on_topic_count += 1
        if is_dataset:
            dataset_count += 1
        if on_topic and is_dataset:
            both_count += 1

        rows.append({
            "pmid": metadata.get("pmid"),
            "title": title,
            "on_topic": on_topic,
            "presents_dataset": is_dataset,
        })

    sampled = len(rows)
    return {
        "sampled": sampled,
        "total_results": len(ids),
        "random_seed": random_seed,
        "on_topic": on_topic_count,
        "presents_dataset": dataset_count,
        "on_topic_and_dataset": both_count,
        "estimated_precision": (both_count / sampled) if sampled else None,
        "samples": rows,
    }


async def build_topic_query(
    topic: str,
    must_include: Sequence[str] = (),
    must_exclude: Sequence[str] = (),
    verify_seeds: bool = True,
    keep_rejected_seeds: bool = False,
    start_query: Optional[str] = None,
    audit_sample: int = 0,
    **synthesis_kwargs,
) -> SynthesisResult:
    """
    End-to-end entry point: identifiers in, validated topic query out.

    Steps: resolve identifiers -> verify seeds against the topic -> synthesize and
    hillclimb the query -> optionally audit the precision of the result set.
    """
    unresolved: List[str] = []

    positive_seeds, unresolved_pos = resolve_seed_papers(must_include, "positive")
    negative_seeds, unresolved_neg = resolve_seed_papers(must_exclude, "negative")
    unresolved.extend(unresolved_pos)
    unresolved.extend(unresolved_neg)

    logger.info(
        f"Resolved {len(positive_seeds)} must-include and {len(negative_seeds)} must-exclude paper(s)."
    )

    if verify_seeds and (positive_seeds or negative_seeds):
        await verify_seed_papers(topic, positive_seeds + negative_seeds)

    accepted_positive = [s for s in positive_seeds if s.accepted or keep_rejected_seeds]
    rejected = [s for s in positive_seeds if not s.accepted]
    if rejected and keep_rejected_seeds:
        logger.warning(f"Keeping {len(rejected)} seed(s) that failed verification (keep_rejected_seeds=True).")

    start_spec = QuerySpec.from_query(start_query) if start_query else None

    result = await synthesize_query(
        topic=topic,
        positive_seeds=accepted_positive,
        negative_seeds=negative_seeds,
        start_spec=start_spec,
        unresolved=unresolved,
        **synthesis_kwargs,
    )

    # Report on rejected seeds even though they did not anchor the search.
    known_pmids = {s.pmid for s in result.seeds}
    result.seeds.extend(s for s in rejected if s.pmid not in known_pmids)

    if audit_sample > 0:
        logger.info(f"Auditing precision on a random sample of {audit_sample} results...")
        result.audit = await audit_query_precision(topic, result.query, sample_size=audit_sample)

    return result
