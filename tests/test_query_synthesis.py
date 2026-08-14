"""
Tests for the query synthesis hillclimb.

The unit tests run the real loop against a simulated PubMed and stubbed agents, so the
control flow (diagnose -> repair -> re-measure -> converge) is exercised without network
or LLM access. The integration tests at the bottom hit live PubMed / a live LLM.
"""

import asyncio

import pytest

from radiology_dataset_db import query_synthesis
from radiology_dataset_db.query_spec import QuerySpec, parse_query_blocks
from radiology_dataset_db.query_synthesis import (
    SeedPaper, diagnose_missing_seeds, evaluate_spec, ground_terms, prune_step,
    synthesize_query)
from radiology_dataset_db.query_synthesis_llm import ProposedQueryTerms, ProposedTerms


# -----------------------------
# SIMULATED PUBMED
# -----------------------------
class FakePubMed:
    """
    A tiny boolean search engine over an in-memory corpus.

    Each paper is a set of terms. A query matches a paper if every AND block shares at
    least one term with it and no NOT block term appears in it -- the same semantics
    PubMed applies to the queries this module builds.
    """

    def __init__(self, papers):
        self.papers = {str(pmid): {t.casefold() for t in terms} for pmid, terms in papers.items()}

    def matching(self, query):
        blocks = parse_query_blocks(query)
        if not blocks:
            return set()

        matched = set()
        for pmid, terms in self.papers.items():
            ok = True
            for block in blocks:
                shares = any(t.casefold() in terms for t in block.terms)
                if block.op == "NOT":
                    if shares:
                        ok = False
                        break
                elif not shares:
                    ok = False
                    break
            if ok:
                matched.add(pmid)
        return matched

    def count(self, query):
        return len(self.matching(query))


@pytest.fixture
def fake_pubmed(monkeypatch):
    """Install a FakePubMed and stub out MeSH validation; returns a setter."""
    holder = {}

    def install(papers):
        engine = FakePubMed(papers)
        holder["engine"] = engine
        monkeypatch.setattr(query_synthesis, "cached_count", lambda query: engine.count(query))
        monkeypatch.setattr(
            query_synthesis,
            "pmids_matching_query",
            lambda pmids, query, **kwargs: engine.matching(query) & {str(p) for p in pmids},
        )
        monkeypatch.setattr(query_synthesis, "mesh_term_exists", lambda term, **kwargs: True)
        return engine

    return install


def stub_agents(monkeypatch, *, proposal=None, recovery=None, exclusion=None, broadening=None):
    """Replace the LLM wrappers with deterministic stubs, recording their calls."""
    calls = {"proposal": 0, "recovery": 0, "exclusion": 0, "broadening": 0}

    async def _proposal(topic, seed_summaries):
        calls["proposal"] += 1
        return proposal

    async def _recovery(**kwargs):
        calls["recovery"] += 1
        return recovery(**kwargs) if callable(recovery) else recovery

    async def _exclusion(**kwargs):
        calls["exclusion"] += 1
        return exclusion(**kwargs) if callable(exclusion) else exclusion

    async def _broadening(**kwargs):
        calls["broadening"] += 1
        return broadening(**kwargs) if callable(broadening) else broadening

    monkeypatch.setattr(query_synthesis, "propose_query_terms", _proposal)
    monkeypatch.setattr(query_synthesis, "propose_recovery_terms", _recovery)
    monkeypatch.setattr(query_synthesis, "propose_exclusion_terms", _exclusion)
    monkeypatch.setattr(query_synthesis, "propose_broadening_terms", _broadening)
    return calls


def seed(pmid, polarity="positive", mesh=()):
    return SeedPaper(
        identifier=str(pmid), polarity=polarity, pmid=str(pmid),
        title=f"Paper {pmid}", abstract="abstract", mesh_terms=list(mesh),
    )


def filler(prefix, n, terms):
    """n throwaway papers sharing the same term set, to give a query realistic bulk."""
    return {f"{prefix}{i}": set(terms) for i in range(n)}


# -----------------------------
# EVALUATION AND DIAGNOSIS
# -----------------------------
def test_evaluate_spec_reports_seed_membership(fake_pubmed):
    fake_pubmed({
        "1": {"dataset[ti]", "ct[tiab]"},
        "2": {"dataset[ti]", "mri[tiab]"},
        "3": {"dataset[ti]", "ct[tiab]", "mice[tiab]"},
    })
    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"])

    evaluation = evaluate_spec(spec, ["1", "2"], ["3"])

    assert evaluation.count == 2
    assert evaluation.found_positive == ["1"]
    assert evaluation.missing_positive == ["2"]
    assert evaluation.caught_negative == ["3"]
    assert evaluation.seed_recall == 0.5
    assert not evaluation.seeds_satisfied


def test_diagnose_identifies_the_failing_block(fake_pubmed):
    fake_pubmed({
        "1": {"ct[tiab]"},                               # fails the dataset block
        "2": {"dataset[ti]"},                            # fails the topic block
        "3": {"dataset[ti]", "ct[tiab]", "mice[tiab]"},  # caught by the exclusion block
    })
    spec = QuerySpec(
        dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"], exclude_terms=["mice[tiab]"],
    )

    diagnosis = diagnose_missing_seeds(spec, ["1", "2", "3"])

    assert diagnosis["1"] == ["dataset"]
    assert diagnosis["2"] == ["topic"]
    assert diagnosis["3"] == ["exclude"]


# -----------------------------
# PRUNING
# -----------------------------
def test_prune_removes_dead_terms_even_when_under_budget(fake_pubmed):
    fake_pubmed({"1": {"dataset[ti]", "ct[tiab]"}})
    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]", "unused[tiab]"])

    result = prune_step(spec, ["1"], max_results=10_000, min_results=0)

    assert result is not None
    new_spec, label, hits_removed = result
    assert label == "topic:unused[tiab]"
    assert hits_removed == 0
    assert new_spec.topic_terms == ["ct[tiab]"]


def test_prune_drops_the_biggest_contributor_that_costs_no_seed(fake_pubmed):
    papers = {"1": {"dataset[ti]", "ct[tiab]"}}
    papers.update(filler("bulk", 400, {"dataset[ti]", "overbroad[tiab]"}))
    papers.update(filler("small", 5, {"dataset[ti]", "mri[tiab]"}))
    fake_pubmed(papers)

    spec = QuerySpec(
        dataset_terms=["dataset[ti]"],
        topic_terms=["ct[tiab]", "mri[tiab]", "overbroad[tiab]"],
    )

    result = prune_step(spec, ["1"], max_results=100, min_results=0)

    assert result is not None
    new_spec, label, hits_removed = result
    assert label == "topic:overbroad[tiab]"
    assert hits_removed == 400
    assert new_spec.topic_terms == ["ct[tiab]", "mri[tiab]"]


def test_prune_refuses_to_drop_a_term_a_seed_depends_on(fake_pubmed):
    papers = {"1": {"dataset[ti]", "onlyterm[tiab]"}}
    papers.update(filler("bulk", 400, {"dataset[ti]", "onlyterm[tiab]"}))
    fake_pubmed(papers)

    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["onlyterm[tiab]", "other[tiab]"])

    result = prune_step(spec, ["1"], max_results=10, min_results=0)

    # 'other[tiab]' is dead so it goes first; 'onlyterm[tiab]' must never be dropped.
    assert result is not None
    assert result[1] == "topic:other[tiab]"
    assert prune_step(result[0], ["1"], max_results=10, min_results=0) is None


def test_prune_is_not_vetoed_by_an_unreachable_seed(fake_pubmed):
    """A must-include paper the query can never retrieve must not freeze all pruning."""
    papers = {"1": {"dataset[ti]", "ct[tiab]"}}
    papers.update(filler("bulk", 400, {"dataset[ti]", "overbroad[tiab]"}))
    fake_pubmed(papers)

    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]", "overbroad[tiab]"])

    # PMID 2 is not in the corpus at all.
    result = prune_step(spec, ["1", "2"], max_results=100, min_results=0)

    assert result is not None
    assert result[1] == "topic:overbroad[tiab]"
    assert result[0].topic_terms == ["ct[tiab]"]  # PMID 1 is still protected


def test_prune_respects_the_result_floor(fake_pubmed):
    papers = {"1": {"dataset[ti]", "ct[tiab]"}}
    papers.update(filler("bulk", 400, {"dataset[ti]", "wide[tiab]"}))
    fake_pubmed(papers)

    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]", "wide[tiab]"])

    # Dropping 'wide[tiab]' would leave 1 result, far below the floor, so it is refused.
    assert prune_step(spec, ["1"], max_results=10, min_results=300) is None


# -----------------------------
# GROUNDING
# -----------------------------
def test_ground_terms_downgrades_hallucinated_mesh_descriptors(monkeypatch):
    monkeypatch.setattr(
        query_synthesis, "mesh_term_exists",
        lambda term, **kwargs: term == "Radiography",
    )

    grounded, notes = ground_terms([
        "Radiography[MeSH]",              # real -> kept
        '"Chest X-Ray Datasets"[MeSH]',   # not a descriptor -> downgraded to [tiab]
        "ct[tiab] OR mri[tiab]",          # malformed -> discarded
    ])

    assert grounded == ["Radiography[MeSH]", '"Chest X-Ray Datasets"[tiab]']
    assert any("not a real MeSH descriptor" in note for note in notes)
    assert any("discarded malformed term" in note for note in notes)


# -----------------------------
# THE LOOP
# -----------------------------
def test_loop_recovers_a_missing_must_include_paper(fake_pubmed, monkeypatch):
    papers = {
        "1": {"dataset[ti]", "ct[tiab]"},
        "2": {"dataset[ti]", "mri[tiab]"},  # only reachable once mri[tiab] is added
    }
    papers.update(filler("bulk", 50, {"dataset[ti]", "ct[tiab]"}))
    fake_pubmed(papers)

    calls = stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"]),
        recovery=ProposedTerms(terms=["mri[tiab]"]),
    )

    result = asyncio.run(synthesize_query(
        topic="ct and mri datasets",
        positive_seeds=[seed("1"), seed("2")],
        min_results=0, max_results=10_000,
    ))

    assert result.converged
    assert result.evaluation.seed_recall == 1.0
    assert "mri[tiab]" in result.spec.topic_terms
    assert calls["recovery"] == 1
    assert [record.action for record in result.history] == ["propose", "recover"]


def test_loop_falls_back_to_indexed_mesh_when_the_llm_proposal_fails(fake_pubmed, monkeypatch):
    papers = {
        "1": {"dataset[ti]", "ct[tiab]"},
        "2": {"dataset[ti]", "Thorax[MeSH]"},
    }
    fake_pubmed(papers)

    # The agent proposes a term that does not actually retrieve the missing paper.
    stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"]),
        recovery=ProposedTerms(terms=["irrelevant[tiab]"]),
    )

    result = asyncio.run(synthesize_query(
        topic="chest datasets",
        positive_seeds=[seed("1"), seed("2", mesh=["Thorax"])],
        min_results=0, max_results=10_000,
    ))

    assert result.converged
    assert result.evaluation.seed_recall == 1.0
    # Recovered using the paper's own indexed MeSH term, not the agent's suggestion.
    assert "Thorax[MeSH]" in result.spec.topic_terms
    assert "irrelevant[tiab]" not in result.spec.topic_terms


def test_loop_drops_an_exclusion_term_that_filters_out_a_required_paper(fake_pubmed, monkeypatch):
    papers = {"1": {"dataset[ti]", "ct[tiab]", "mice[tiab]"}}
    fake_pubmed(papers)

    stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(
            dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"], exclude_terms=["mice[tiab]"],
        ),
    )

    result = asyncio.run(synthesize_query(
        topic="ct datasets", positive_seeds=[seed("1")], min_results=0, max_results=10_000,
    ))

    assert result.converged
    assert result.spec.exclude_terms == []
    assert result.evaluation.seed_recall == 1.0


def test_loop_excludes_a_must_exclude_paper(fake_pubmed, monkeypatch):
    papers = {
        "1": {"dataset[ti]", "ct[tiab]"},
        "99": {"dataset[ti]", "ct[tiab]", "crystallography[tiab]"},
    }
    fake_pubmed(papers)

    stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"]),
        exclusion=ProposedTerms(terms=["crystallography[tiab]"]),
    )

    result = asyncio.run(synthesize_query(
        topic="ct datasets",
        positive_seeds=[seed("1")],
        negative_seeds=[seed("99", polarity="negative")],
        min_results=0, max_results=10_000,
    ))

    assert result.converged
    assert result.spec.exclude_terms == ["crystallography[tiab]"]
    assert result.evaluation.caught_negative == []
    assert result.evaluation.seed_recall == 1.0


def test_loop_rejects_an_exclusion_term_that_would_lose_a_required_paper(fake_pubmed, monkeypatch):
    # The only discriminator the agent can find also appears in a must-include paper.
    papers = {
        "1": {"dataset[ti]", "ct[tiab]", "shared[tiab]"},
        "99": {"dataset[ti]", "ct[tiab]", "shared[tiab]"},
    }
    fake_pubmed(papers)

    stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"]),
        exclusion=ProposedTerms(terms=["shared[tiab]"]),
    )

    result = asyncio.run(synthesize_query(
        topic="ct datasets",
        positive_seeds=[seed("1")],
        negative_seeds=[seed("99", polarity="negative")],
        min_results=0, max_results=10_000,
    ))

    # Recall wins: the unwanted paper stays rather than the required one being dropped.
    assert result.spec.exclude_terms == []
    assert result.evaluation.seed_recall == 1.0


def test_loop_prunes_until_the_query_fits_the_budget(fake_pubmed, monkeypatch):
    papers = {"1": {"dataset[ti]", "ct[tiab]"}}
    papers.update(filler("wide", 500, {"dataset[ti]", "overbroad[tiab]"}))
    papers.update(filler("also", 300, {"dataset[ti]", "alsowide[tiab]"}))
    fake_pubmed(papers)

    stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(
            dataset_terms=["dataset[ti]"],
            topic_terms=["ct[tiab]", "overbroad[tiab]", "alsowide[tiab]"],
        ),
    )

    result = asyncio.run(synthesize_query(
        topic="ct datasets", positive_seeds=[seed("1")], min_results=0, max_results=100,
    ))

    assert result.converged
    assert result.evaluation.count <= 100
    assert result.evaluation.seed_recall == 1.0
    assert result.spec.topic_terms == ["ct[tiab]"]
    assert [record.action for record in result.history].count("prune") == 2


def test_loop_broadens_an_implausibly_narrow_query(fake_pubmed, monkeypatch):
    papers = {"1": {"dataset[ti]", "ct[tiab]"}}
    papers.update(filler("more", 200, {"dataset[ti]", "tomography[tiab]"}))
    fake_pubmed(papers)

    calls = stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"]),
        broadening=ProposedTerms(terms=["tomography[tiab]"]),
    )

    result = asyncio.run(synthesize_query(
        topic="ct datasets", positive_seeds=[seed("1")], min_results=50, max_results=10_000,
    ))

    assert calls["broadening"] == 1
    assert "tomography[tiab]" in result.spec.topic_terms
    assert result.evaluation.count >= 50


def test_loop_reports_when_a_seed_cannot_be_recovered(fake_pubmed, monkeypatch):
    # PMID 2 is not in the corpus at all, so no term can ever retrieve it. The two
    # over-broad terms keep the loop running for further iterations after the failure.
    papers = {"1": {"dataset[ti]", "ct[tiab]"}}
    papers.update(filler("wide", 500, {"dataset[ti]", "overbroad[tiab]"}))
    papers.update(filler("also", 300, {"dataset[ti]", "alsowide[tiab]"}))
    fake_pubmed(papers)

    calls = stub_agents(
        monkeypatch,
        proposal=ProposedQueryTerms(
            dataset_terms=["dataset[ti]"],
            topic_terms=["ct[tiab]", "overbroad[tiab]", "alsowide[tiab]"],
        ),
        recovery=ProposedTerms(terms=["nothing[tiab]"]),
    )

    result = asyncio.run(synthesize_query(
        topic="ct datasets",
        positive_seeds=[seed("1"), seed("2")],
        min_results=0, max_results=100, max_iterations=6,
    ))

    assert result.evaluation.missing_positive == ["2"]
    assert result.evaluation.seed_recall == 0.5
    assert result.to_dict()["missing_positive_seeds"] == ["2"]
    # A fixed point that leaves a required paper out is a stall, not convergence.
    assert not result.converged
    # The loop kept working (pruning) after giving up on the seed...
    assert [record.action for record in result.history].count("prune") == 2
    # ...but the hopeless seed was tried once per failing block and never retried,
    # rather than burning two agent calls on every subsequent iteration.
    assert calls["recovery"] == 2


def test_loop_can_start_from_an_existing_query(fake_pubmed, monkeypatch):
    fake_pubmed({"1": {"dataset[ti]", "ct[tiab]"}})
    calls = stub_agents(monkeypatch, proposal=None)

    result = asyncio.run(synthesize_query(
        topic="ct datasets",
        positive_seeds=[seed("1")],
        start_spec=QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["ct[tiab]"]),
        min_results=0, max_results=10_000,
    ))

    assert calls["proposal"] == 0  # no proposal agent call when seeded with a query
    assert result.converged
    assert result.query == "(dataset[ti]) AND (ct[tiab])"


# -----------------------------
# INTEGRATION (live PubMed + live LLM)
# -----------------------------
@pytest.mark.integration
def test_hand_written_radiology_query_retrieves_its_ground_truth():
    """The hand-tuned query is the benchmark the synthesized one is compared against."""
    from radiology_dataset_db.config import PUBMED_QUERY_RADIOLOGY
    from tests.conftest import _radiology_paper_ground_truth

    spec = QuerySpec.from_query(PUBMED_QUERY_RADIOLOGY)
    pmids = [p["pmid"] for p in _radiology_paper_ground_truth().values() if p.get("pmid")]

    evaluation = evaluate_spec(spec, pmids, [])

    assert evaluation.count > 0
    assert evaluation.missing_positive == [], f"query lost ground-truth papers: {evaluation.missing_positive}"


@pytest.mark.integration
@pytest.mark.slow
def test_synthesize_query_end_to_end():
    """Full loop against live PubMed and a live LLM. Requires a running model backend."""
    from radiology_dataset_db.query_synthesis import build_topic_query

    result = asyncio.run(build_topic_query(
        topic="chest x-ray datasets",
        must_include=["31831740"],  # MIMIC-CXR
        max_results=8000,
        min_results=50,
        max_iterations=8,
    ))

    assert result.query
    assert result.evaluation.count > 0
    assert "31831740" in result.evaluation.found_positive
