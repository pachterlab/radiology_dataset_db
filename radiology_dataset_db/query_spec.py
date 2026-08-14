"""
Structured representation of a PubMed boolean query.

This module is deliberately dependency-free (no Entrez, no LLM) so that it can be
imported from ``pubmed_utils.py`` without creating a circular import. Anything that
needs to actually hit PubMed lives in ``pubmed_utils.py`` / ``query_synthesis.py``.

A synthesized query always has the shape::

    (dataset terms) AND (topic terms) NOT (exclusion terms)

- ``dataset terms``  -- "this paper introduces a dataset" signals (dataset[ti], benchmark[ti], ...)
- ``topic terms``    -- "this paper is about my subject" signals ("Radiography"[MeSH], MRI[tiab], ...)
- ``exclusion terms``-- known false-positive traps (NMR[tiab], crystallograph*[tiab], ...)

The parser is tolerant: it handles arbitrarily nested parentheses, quoted phrases
containing boolean keywords, lowercase operators, and the ``NOT (... OR (...))``
nesting used by the hand-written radiology query.
"""

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

# Characters that may appear inside a search term. Used for operator word-boundary
# checks so that e.g. "Andrology[tiab]" is not mistaken for an AND operator.
_WORD_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_-*"
)

BLOCK_NAMES = ("dataset", "topic", "exclude")

# Human-facing labels used by drop_one_analysis output
_BLOCK_LABELS = ("LEFT", "RIGHT", "EXCLUDE")


# -----------------------------
# LOW-LEVEL PARSING
# -----------------------------
def _top_level_operator_positions(text: str, operators: Sequence[str]) -> List[Tuple[int, int, str]]:
    """
    Find boolean operators that appear at parenthesis depth 0 and outside quotes.

    Returns a list of (start, end, operator) tuples, where operator is upper-cased.
    """
    positions: List[Tuple[int, int, str]] = []
    upper = text.upper()
    depth = 0
    in_quotes = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if in_quotes:
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(depth - 1, 0)
            i += 1
            continue

        if depth == 0:
            matched = None
            for op in operators:
                end = i + len(op)
                if upper[i:end] != op:
                    continue
                before_ok = i == 0 or text[i - 1] not in _WORD_CHARS
                after_ok = end >= n or text[end] not in _WORD_CHARS
                if before_ok and after_ok:
                    matched = (i, end, op)
                    break
            if matched is not None:
                positions.append(matched)
                i = matched[1]
                continue

        i += 1

    return positions


def _split_on_operators(text: str, operators: Sequence[str]) -> List[Tuple[Optional[str], str]]:
    """Split text at top-level operators, returning [(preceding_operator, segment), ...]."""
    positions = _top_level_operator_positions(text, operators)
    segments: List[Tuple[Optional[str], str]] = []
    prev_end = 0
    prev_op: Optional[str] = None

    for start, end, op in positions:
        segments.append((prev_op, text[prev_end:start]))
        prev_op = op
        prev_end = end

    segments.append((prev_op, text[prev_end:]))
    return segments


def _strip_outer_parens(text: str) -> str:
    """Remove parentheses that wrap the entire expression, e.g. '(a OR b)' -> 'a OR b'."""
    text = text.strip()

    while len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        depth = 0
        in_quotes = False
        wraps_whole = True

        for i, ch in enumerate(text):
            if ch == '"':
                in_quotes = not in_quotes
                continue
            if in_quotes:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    wraps_whole = False
                    break

        if not wraps_whole or depth != 0:
            break
        text = text[1:-1].strip()

    return text


def split_or_terms(segment: str) -> List[str]:
    """
    Flatten a segment into its OR-ed terms.

    Nested OR groups are flattened (OR is associative), but a parenthesized group
    containing a top-level AND/NOT is kept intact as a single atomic term so that
    its meaning is never silently changed.
    """
    segment = _strip_outer_parens(segment)
    if not segment:
        return []

    terms: List[str] = []
    for _, part in _split_on_operators(segment, ("OR",)):
        part = part.strip()
        if not part:
            continue

        inner = _strip_outer_parens(part)
        is_parenthesized = inner != part.strip()
        has_nested_boolean = bool(_top_level_operator_positions(inner, ("AND", "NOT")))

        if is_parenthesized and not has_nested_boolean:
            terms.extend(split_or_terms(inner))
        else:
            terms.append(part)

    return terms


@dataclass
class QueryBlock:
    """One parenthesized OR-group, joined to the previous block by ``op``."""
    op: str  # "AND" or "NOT"; the first block is "AND" by convention
    terms: List[str] = field(default_factory=list)


def parse_query_blocks(query: str) -> List[QueryBlock]:
    """
    Parse an arbitrary PubMed boolean query into its top-level AND/NOT blocks.

    Unlike a naive ``query.split("AND")``, this handles NOT clauses, nested
    parentheses, and quoted phrases.
    """
    query = " ".join((query or "").split())
    if not query:
        return []

    blocks: List[QueryBlock] = []
    for op, segment in _split_on_operators(query, ("AND", "NOT")):
        terms = split_or_terms(segment)
        if not terms:
            continue
        blocks.append(QueryBlock(op=(op or "AND").upper(), terms=terms))

    return blocks


def render_query_blocks(blocks: Sequence[QueryBlock]) -> str:
    """Render blocks back into a PubMed query string."""
    parts: List[str] = []
    for block in blocks:
        if not block.terms:
            continue
        rendered = "(" + " OR ".join(block.terms) + ")"
        if not parts:
            parts.append(rendered)
        else:
            parts.append(f"{block.op} {rendered}")
    return " ".join(parts)


# -----------------------------
# TERM HELPERS
# -----------------------------
_TERM_PATTERN = re.compile(r'^\s*(?:"(?P<quoted>[^"]+)"|(?P<bare>[^\[\]]+?))\s*\[(?P<field>[^\[\]]+)\]\s*$')


def split_term(term: str) -> Tuple[str, Optional[str]]:
    """
    Split a search term into (text, field).

    '"Radiology Information Systems"[MeSH]' -> ('Radiology Information Systems', 'MeSH')
    'radiograph[tiab]'                      -> ('radiograph', 'tiab')
    'radiograph'                            -> ('radiograph', None)
    """
    match = _TERM_PATTERN.match(term or "")
    if not match:
        return (term or "").strip().strip('"'), None
    text = match.group("quoted") or match.group("bare") or ""
    return text.strip(), match.group("field").strip()


def make_term(text: str, field_tag: Optional[str]) -> str:
    """Rebuild a search term, quoting multi-word text."""
    text = (text or "").strip().strip('"')
    if not text:
        return ""
    rendered = f'"{text}"' if " " in text else text
    return f"{rendered}[{field_tag}]" if field_tag else rendered


def is_mesh_term(term: str) -> bool:
    _, field_tag = split_term(term)
    return bool(field_tag) and field_tag.strip().lower().startswith("mesh")


def normalize_term(term: str) -> str:
    """Collapse whitespace so that terms compare and render consistently."""
    return " ".join((term or "").split())


def _dedupe_terms(terms: Sequence[str]) -> List[str]:
    """Drop empties and case-insensitive duplicates, preserving order."""
    seen = set()
    result: List[str] = []
    for term in terms:
        term = normalize_term(term)
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


# -----------------------------
# QUERY SPEC
# -----------------------------
@dataclass
class QuerySpec:
    """A three-block query: (dataset) AND (topic) NOT (exclude)."""

    dataset_terms: List[str] = field(default_factory=list)
    topic_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)

    # -- block access ------------------------------------------------------
    def block(self, name: str) -> List[str]:
        if name not in BLOCK_NAMES:
            raise ValueError(f"Unknown block {name!r}. Expected one of {BLOCK_NAMES}.")
        return getattr(self, f"{name}_terms")

    def set_block(self, name: str, terms: Sequence[str]) -> None:
        if name not in BLOCK_NAMES:
            raise ValueError(f"Unknown block {name!r}. Expected one of {BLOCK_NAMES}.")
        setattr(self, f"{name}_terms", _dedupe_terms(terms))

    def iter_terms(self) -> Iterator[Tuple[str, str]]:
        """Yield (block_name, term) for every term in the spec."""
        for name in BLOCK_NAMES:
            for term in self.block(name):
                yield name, term

    def num_terms(self) -> int:
        return sum(len(self.block(name)) for name in BLOCK_NAMES)

    # -- construction ------------------------------------------------------
    def copy(self) -> "QuerySpec":
        return QuerySpec(
            dataset_terms=list(self.dataset_terms),
            topic_terms=list(self.topic_terms),
            exclude_terms=list(self.exclude_terms),
        )

    def normalized(self) -> "QuerySpec":
        spec = QuerySpec()
        for name in BLOCK_NAMES:
            spec.set_block(name, self.block(name))
        return spec

    def with_terms_added(self, name: str, terms: Sequence[str]) -> "QuerySpec":
        spec = self.copy()
        spec.set_block(name, list(spec.block(name)) + list(terms))
        return spec

    def with_term_removed(self, name: str, term: str) -> "QuerySpec":
        spec = self.copy()
        key = normalize_term(term).casefold()
        spec.set_block(name, [t for t in spec.block(name) if normalize_term(t).casefold() != key])
        return spec

    # -- rendering ---------------------------------------------------------
    def blocks(self) -> List[QueryBlock]:
        blocks: List[QueryBlock] = []
        if self.dataset_terms:
            blocks.append(QueryBlock(op="AND", terms=list(self.dataset_terms)))
        if self.topic_terms:
            blocks.append(QueryBlock(op="AND", terms=list(self.topic_terms)))
        if self.exclude_terms:
            blocks.append(QueryBlock(op="NOT", terms=list(self.exclude_terms)))
        return blocks

    def render(self) -> str:
        return render_query_blocks(self.blocks())

    def block_query(self, name: str) -> str:
        """Render a single block on its own, e.g. '(a OR b)'. Empty block -> ''."""
        terms = self.block(name)
        if not terms:
            return ""
        return "(" + " OR ".join(terms) + ")"

    def is_runnable(self) -> bool:
        """A query with no positive block would match all of PubMed (or nothing)."""
        return bool(self.dataset_terms or self.topic_terms)

    def to_dict(self) -> dict:
        return {
            "dataset_terms": list(self.dataset_terms),
            "topic_terms": list(self.topic_terms),
            "exclude_terms": list(self.exclude_terms),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuerySpec":
        spec = cls()
        for name in BLOCK_NAMES:
            spec.set_block(name, data.get(f"{name}_terms", []) or [])
        return spec

    @classmethod
    def from_query(cls, query: str) -> "QuerySpec":
        """
        Parse an existing query string into a QuerySpec.

        Raises ValueError if the query has more than two AND-blocks, since silently
        merging them would change the query's meaning.
        """
        blocks = parse_query_blocks(query)
        and_blocks = [b for b in blocks if b.op == "AND"]
        not_blocks = [b for b in blocks if b.op == "NOT"]

        if len(and_blocks) > 2:
            raise ValueError(
                f"Expected at most two AND-blocks in the shape "
                f"'(dataset) AND (topic) NOT (exclude)', found {len(and_blocks)}. "
                "Rewrite the query or build the QuerySpec explicitly."
            )

        spec = cls()
        if len(and_blocks) >= 1:
            spec.set_block("dataset", and_blocks[0].terms)
        if len(and_blocks) >= 2:
            spec.set_block("topic", and_blocks[1].terms)
        if not_blocks:
            spec.set_block("exclude", [t for b in not_blocks for t in b.terms])
        return spec

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.render()
