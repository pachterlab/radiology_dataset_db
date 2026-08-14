"""
LLM agents used by the PubMed query synthesis loop.

Each agent is small and single-purpose; the loop in ``query_synthesis.py`` decides when
to call which one based on measured PubMed counts, and validates every term the LLM
proposes before it goes into a query.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from radiology_dataset_db.config import (
    LOG_LEVEL, MODEL,
    QUERY_BROADENING_AGENT_INSTRUCTIONS, QUERY_BROADENING_INSTRUCTIONS,
    QUERY_EXCLUSION_AGENT_INSTRUCTIONS, QUERY_EXCLUSION_INSTRUCTIONS,
    QUERY_PROPOSAL_AGENT_INSTRUCTIONS, QUERY_PROPOSAL_INSTRUCTIONS,
    QUERY_RECOVERY_AGENT_INSTRUCTIONS, QUERY_RECOVERY_INSTRUCTIONS,
    SEED_VERIFICATION_AGENT_INSTRUCTIONS, SEED_VERIFICATION_INSTRUCTIONS,
    TOPIC_RELEVANCE_AGENT_INSTRUCTIONS, TOPIC_RELEVANCE_INSTRUCTIONS)

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

MAX_ABSTRACT_CHARS = 4000


def _truncate(text: Optional[str], limit: int = MAX_ABSTRACT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " [...truncated]"


def _format_terms(terms: Sequence[str]) -> str:
    return ", ".join(terms) if terms else "(none)"


# -----------------------------
# SCHEMAS
# -----------------------------
class SeedVerification(BaseModel):
    is_on_topic: bool
    presents_dataset: bool
    reason: str = ""


class ProposedQueryTerms(BaseModel):
    dataset_terms: List[str] = Field(default_factory=list)
    topic_terms: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)


class ProposedTerms(BaseModel):
    terms: List[str] = Field(default_factory=list)
    reason: str = ""


class TopicRelevance(BaseModel):
    is_on_topic: bool


# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class SeedVerificationDeps:
    topic: str
    title: str
    abstract: str


@dataclass
class QueryProposalDeps:
    topic: str
    seed_summaries: str


@dataclass
class RecoveryDeps:
    topic: str
    block_name: str
    block_terms: str
    query: str
    title: str
    abstract: str
    mesh_terms: str


@dataclass
class ExclusionDeps:
    topic: str
    query: str
    title: str
    abstract: str
    mesh_terms: str
    positive_summaries: str


@dataclass
class BroadeningDeps:
    topic: str
    query: str
    topic_terms: str
    current_count: int
    target_count: int


@dataclass
class TopicRelevanceDeps:
    topic: str
    title: str
    abstract: str


# -----------------------------
# AGENTS
# -----------------------------
seed_verification_agent = Agent(
    MODEL,
    deps_type=SeedVerificationDeps,
    output_type=SeedVerification,
    instructions=SEED_VERIFICATION_INSTRUCTIONS,
)


@seed_verification_agent.instructions
async def _seed_verification_text(ctx: RunContext[SeedVerificationDeps]) -> str:
    return f"""
Topic: {ctx.deps.topic}

Paper title: {ctx.deps.title}
Paper abstract: {_truncate(ctx.deps.abstract)}
"""


query_proposal_agent = Agent(
    MODEL,
    deps_type=QueryProposalDeps,
    output_type=ProposedQueryTerms,
    instructions=QUERY_PROPOSAL_INSTRUCTIONS,
)


@query_proposal_agent.instructions
async def _query_proposal_text(ctx: RunContext[QueryProposalDeps]) -> str:
    return f"""
Topic: {ctx.deps.topic}

Example papers that MUST be retrieved by the query (use their vocabulary and indexed
MeSH terms as evidence for which terms actually work):
{ctx.deps.seed_summaries}
"""


recovery_agent = Agent(
    MODEL,
    deps_type=RecoveryDeps,
    output_type=ProposedTerms,
    instructions=QUERY_RECOVERY_INSTRUCTIONS,
)


@recovery_agent.instructions
async def _recovery_text(ctx: RunContext[RecoveryDeps]) -> str:
    return f"""
Topic: {ctx.deps.topic}
Current query: {ctx.deps.query}

The paper below fails to match the '{ctx.deps.block_name}' block, which currently contains:
{ctx.deps.block_terms}

Missing paper title: {ctx.deps.title}
Missing paper abstract: {_truncate(ctx.deps.abstract)}
Missing paper indexed MeSH terms: {ctx.deps.mesh_terms}

Propose terms to add to the '{ctx.deps.block_name}' block so that this paper matches.
"""


exclusion_agent = Agent(
    MODEL,
    deps_type=ExclusionDeps,
    output_type=ProposedTerms,
    instructions=QUERY_EXCLUSION_INSTRUCTIONS,
)


@exclusion_agent.instructions
async def _exclusion_text(ctx: RunContext[ExclusionDeps]) -> str:
    return f"""
Topic: {ctx.deps.topic}
Current query: {ctx.deps.query}

Paper that must be EXCLUDED:
Title: {ctx.deps.title}
Abstract: {_truncate(ctx.deps.abstract)}
Indexed MeSH terms: {ctx.deps.mesh_terms}

Papers that must REMAIN in the results (your exclusion terms must not match these):
{ctx.deps.positive_summaries}
"""


broadening_agent = Agent(
    MODEL,
    deps_type=BroadeningDeps,
    output_type=ProposedTerms,
    instructions=QUERY_BROADENING_INSTRUCTIONS,
)


@broadening_agent.instructions
async def _broadening_text(ctx: RunContext[BroadeningDeps]) -> str:
    return f"""
Topic: {ctx.deps.topic}
Current query: {ctx.deps.query}
Current topic terms: {ctx.deps.topic_terms}

The query currently returns {ctx.deps.current_count} results, but at least
{ctx.deps.target_count} are expected for this topic.
"""


topic_relevance_agent = Agent(
    MODEL,
    deps_type=TopicRelevanceDeps,
    output_type=TopicRelevance,
    instructions=TOPIC_RELEVANCE_INSTRUCTIONS,
)


@topic_relevance_agent.instructions
async def _topic_relevance_text(ctx: RunContext[TopicRelevanceDeps]) -> str:
    return f"""
Topic: {ctx.deps.topic}

Paper title: {ctx.deps.title}
Paper abstract: {_truncate(ctx.deps.abstract)}
"""


# -----------------------------
# PUBLIC WRAPPERS
# -----------------------------
async def verify_seed_paper(topic: str, title: str, abstract: str) -> Optional[SeedVerification]:
    """Check that a user-supplied seed is on topic and actually introduces a dataset."""
    try:
        result = await seed_verification_agent.run(
            SEED_VERIFICATION_AGENT_INSTRUCTIONS,
            deps=SeedVerificationDeps(topic=topic, title=title, abstract=abstract),
        )
        return result.output
    except Exception as e:
        logger.error(f"Seed verification failed for {title!r}: {e}")
        return None


async def propose_query_terms(topic: str, seed_summaries: str) -> Optional[ProposedQueryTerms]:
    """Propose the initial three-block term set for a topic."""
    try:
        result = await query_proposal_agent.run(
            QUERY_PROPOSAL_AGENT_INSTRUCTIONS,
            deps=QueryProposalDeps(topic=topic, seed_summaries=seed_summaries),
        )
        return result.output
    except Exception as e:
        logger.error(f"Query term proposal failed: {e}")
        return None


async def propose_recovery_terms(
    topic: str,
    query: str,
    block_name: str,
    block_terms: Sequence[str],
    title: str,
    abstract: str,
    mesh_terms: Sequence[str],
) -> Optional[ProposedTerms]:
    """Propose terms that would make a missing seed paper match the failing block."""
    try:
        result = await recovery_agent.run(
            QUERY_RECOVERY_AGENT_INSTRUCTIONS,
            deps=RecoveryDeps(
                topic=topic,
                block_name=block_name,
                block_terms=_format_terms(block_terms),
                query=query,
                title=title,
                abstract=abstract,
                mesh_terms=_format_terms(mesh_terms),
            ),
        )
        return result.output
    except Exception as e:
        logger.error(f"Recovery term proposal failed for {title!r}: {e}")
        return None


async def propose_exclusion_terms(
    topic: str,
    query: str,
    title: str,
    abstract: str,
    mesh_terms: Sequence[str],
    positive_summaries: str,
) -> Optional[ProposedTerms]:
    """Propose NOT-block terms that would drop an unwanted paper."""
    try:
        result = await exclusion_agent.run(
            QUERY_EXCLUSION_AGENT_INSTRUCTIONS,
            deps=ExclusionDeps(
                topic=topic,
                query=query,
                title=title,
                abstract=abstract,
                mesh_terms=_format_terms(mesh_terms),
                positive_summaries=positive_summaries,
            ),
        )
        return result.output
    except Exception as e:
        logger.error(f"Exclusion term proposal failed for {title!r}: {e}")
        return None


async def propose_broadening_terms(
    topic: str,
    query: str,
    topic_terms: Sequence[str],
    current_count: int,
    target_count: int,
) -> Optional[ProposedTerms]:
    """Propose additional topic terms when the query is implausibly narrow."""
    try:
        result = await broadening_agent.run(
            QUERY_BROADENING_AGENT_INSTRUCTIONS,
            deps=BroadeningDeps(
                topic=topic,
                query=query,
                topic_terms=_format_terms(topic_terms),
                current_count=current_count,
                target_count=target_count,
            ),
        )
        return result.output
    except Exception as e:
        logger.error(f"Broadening term proposal failed: {e}")
        return None


async def paper_is_on_topic(topic: str, title: str, abstract: str) -> Optional[bool]:
    """Judge topical fit of a sampled paper, for the precision audit."""
    try:
        result = await topic_relevance_agent.run(
            TOPIC_RELEVANCE_AGENT_INSTRUCTIONS,
            deps=TopicRelevanceDeps(topic=topic, title=title, abstract=abstract),
        )
        return result.output.is_on_topic
    except Exception as e:
        logger.error(f"Topic relevance judgement failed for {title!r}: {e}")
        return None
