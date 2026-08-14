"""Unit tests for the query structure core. No network, no LLM."""

import pytest

from radiology_dataset_db.config import PUBMED_QUERY_RADIOLOGY
from radiology_dataset_db.query_spec import (
    QuerySpec, is_mesh_term, make_term, parse_query_blocks, render_query_blocks,
    split_or_terms, split_term)
from radiology_dataset_db.query_synthesis import normalize_proposed_term


# -----------------------------
# PARSING
# -----------------------------
def test_parse_simple_two_block_query():
    blocks = parse_query_blocks("(dataset[ti] OR database[ti]) AND (MRI[tiab] OR CT[tiab])")
    assert [b.op for b in blocks] == ["AND", "AND"]
    assert blocks[0].terms == ["dataset[ti]", "database[ti]"]
    assert blocks[1].terms == ["MRI[tiab]", "CT[tiab]"]


def test_parse_handles_not_block():
    blocks = parse_query_blocks("(dataset[ti]) AND (MRI[tiab]) NOT (mice[tiab] OR mouse[tiab])")
    assert [b.op for b in blocks] == ["AND", "AND", "NOT"]
    assert blocks[2].terms == ["mice[tiab]", "mouse[tiab]"]


def test_parse_flattens_nested_or_groups_and_lowercase_operators():
    query = "(a[ti]) AND (b[tiab]) NOT ((x[tiab] OR y[tiab]) OR (mice[tiab] or mouse[tiab]))"
    blocks = parse_query_blocks(query)
    assert blocks[2].terms == ["x[tiab]", "y[tiab]", "mice[tiab]", "mouse[tiab]"]


def test_parse_keeps_quoted_phrases_intact():
    blocks = parse_query_blocks('("Databases, Factual"[MeSH] OR "data sharing"[ti]) AND (CT[tiab])')
    assert blocks[0].terms == ['"Databases, Factual"[MeSH]', '"data sharing"[ti]']


def test_parse_does_not_split_inside_quoted_operator_words():
    blocks = parse_query_blocks('("imaging AND reporting"[ti] OR x[ti]) AND (CT[tiab])')
    assert blocks[0].terms == ['"imaging AND reporting"[ti]', "x[ti]"]
    assert len(blocks) == 2


def test_parse_does_not_treat_word_prefixes_as_operators():
    # "Andrology" starts with AND; "NOTCH" starts with NOT.
    blocks = parse_query_blocks("(Andrology[tiab] OR NOTCH[tiab]) AND (CT[tiab])")
    assert blocks[0].terms == ["Andrology[tiab]", "NOTCH[tiab]"]
    assert len(blocks) == 2


def test_split_or_terms_on_bare_segment():
    assert split_or_terms("a[ti] OR b[ti]") == ["a[ti]", "b[ti]"]
    assert split_or_terms("(a[ti])") == ["a[ti]"]
    assert split_or_terms("") == []


def test_parenthesized_group_containing_and_is_kept_atomic():
    # Flattening this would change the query's meaning, so it stays as one term.
    terms = split_or_terms("(a[ti] AND b[ti]) OR c[ti]")
    assert terms == ["(a[ti] AND b[ti])", "c[ti]"]


# -----------------------------
# ROUND TRIPPING
# -----------------------------
def test_render_round_trips():
    query = "(dataset[ti] OR database[ti]) AND (MRI[tiab]) NOT (mice[tiab])"
    assert render_query_blocks(parse_query_blocks(query)) == query


def test_real_radiology_query_round_trips_semantically():
    spec = QuerySpec.from_query(PUBMED_QUERY_RADIOLOGY)

    assert '"Database Management Systems"[MeSH]' in spec.dataset_terms
    assert '"Radiology"[MeSH]' in spec.topic_terms
    assert "MRI[tiab]" in spec.topic_terms
    # The NOT clause is nested three groups deep in the hand-written query.
    assert "NMR[tiab]" in spec.exclude_terms
    assert "crystallograph*[tiab]" in spec.exclude_terms
    assert "mouse[tiab]" in spec.exclude_terms

    # Re-parsing the rendered query yields the same spec.
    assert QuerySpec.from_query(spec.render()).to_dict() == spec.to_dict()


def test_from_query_rejects_ambiguous_shapes():
    with pytest.raises(ValueError):
        QuerySpec.from_query("(a[ti]) AND (b[ti]) AND (c[ti])")


# -----------------------------
# QUERYSPEC BEHAVIOUR
# -----------------------------
def test_spec_dedupes_case_insensitively_and_preserves_order():
    spec = QuerySpec()
    spec.set_block("topic", ["CT[tiab]", "ct[TIAB]", "MRI[tiab]", "  ", "MRI[tiab]"])
    assert spec.topic_terms == ["CT[tiab]", "MRI[tiab]"]


def test_spec_add_and_remove_terms_are_immutable():
    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["CT[tiab]"])
    added = spec.with_terms_added("topic", ["MRI[tiab]"])
    removed = added.with_term_removed("topic", "ct[tiab]")

    assert spec.topic_terms == ["CT[tiab]"]  # original untouched
    assert added.topic_terms == ["CT[tiab]", "MRI[tiab]"]
    assert removed.topic_terms == ["MRI[tiab]"]  # removal is case-insensitive


def test_render_skips_empty_blocks():
    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["CT[tiab]"])
    assert spec.render() == "(dataset[ti]) AND (CT[tiab])"

    spec.set_block("exclude", ["mice[tiab]"])
    assert spec.render() == "(dataset[ti]) AND (CT[tiab]) NOT (mice[tiab])"

    topic_only = QuerySpec(topic_terms=["CT[tiab]"])
    assert topic_only.render() == "(CT[tiab])"
    assert topic_only.is_runnable()
    assert not QuerySpec().is_runnable()


def test_block_query_and_iter_terms():
    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["CT[tiab]", "MRI[tiab]"])
    assert spec.block_query("topic") == "(CT[tiab] OR MRI[tiab])"
    assert spec.block_query("exclude") == ""
    assert list(spec.iter_terms()) == [
        ("dataset", "dataset[ti]"),
        ("topic", "CT[tiab]"),
        ("topic", "MRI[tiab]"),
    ]


def test_spec_dict_round_trip():
    spec = QuerySpec(dataset_terms=["dataset[ti]"], topic_terms=["CT[tiab]"], exclude_terms=["mice[tiab]"])
    assert QuerySpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


# -----------------------------
# TERM HELPERS
# -----------------------------
@pytest.mark.parametrize("term, expected", [
    ('"Radiology Information Systems"[MeSH]', ("Radiology Information Systems", "MeSH")),
    ("radiograph[tiab]", ("radiograph", "tiab")),
    ("crystallograph*[tiab]", ("crystallograph*", "tiab")),
    ("radiograph", ("radiograph", None)),
])
def test_split_term(term, expected):
    assert split_term(term) == expected


def test_make_term_quotes_only_when_needed():
    assert make_term("Databases, Factual", "MeSH") == '"Databases, Factual"[MeSH]'
    assert make_term("radiograph", "tiab") == "radiograph[tiab]"
    assert make_term("", "tiab") == ""


def test_is_mesh_term():
    assert is_mesh_term('"Radiography"[MeSH]')
    assert is_mesh_term("Radiography[MeSH Terms]")
    assert not is_mesh_term("radiograph[tiab]")


# -----------------------------
# LLM OUTPUT NORMALIZATION
# -----------------------------
def test_normalize_proposed_term_defaults_untagged_terms_to_tiab():
    assert normalize_proposed_term("chest radiograph") == '"chest radiograph"[tiab]'
    assert normalize_proposed_term("radiograph") == "radiograph[tiab]"


def test_normalize_proposed_term_preserves_valid_tags_and_collapses_whitespace():
    # Redundant quotes around a single word are dropped; PubMed treats the two forms
    # identically, and consistent rendering keeps term de-duplication reliable.
    assert normalize_proposed_term('  "Radiography"[MeSH] ') == "Radiography[MeSH]"
    assert normalize_proposed_term('"Databases, Factual"[MeSH]') == '"Databases, Factual"[MeSH]'
    assert normalize_proposed_term("chest\n  radiograph[tiab]") == '"chest radiograph"[tiab]'


@pytest.mark.parametrize("bad_term", [
    "CT[tiab] OR MRI[tiab]",   # more than one concept
    "(CT[tiab])",              # parenthesized
    "CT[tiab] AND MRI[tiab]",
    "NOT mice[tiab]",
    "",
    "   ",
])
def test_normalize_proposed_term_rejects_unusable_terms(bad_term):
    assert normalize_proposed_term(bad_term) is None
