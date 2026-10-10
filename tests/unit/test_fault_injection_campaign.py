"""The systematic fault campaign (paper §6.6) keeps the behaviour it reports."""

from __future__ import annotations

import pytest

from experiments.run_fault_injection_campaign import FAULTS, run_campaign, summarise


@pytest.fixture(scope="module")
def summary():
    return summarise(run_campaign())


def test_catalogue_covers_both_stages_and_both_fault_levels():
    assert len(FAULTS) == 13
    assert {f.stage for f in FAULTS} == {"classification", "prioritization"}
    assert sum(f.value_range for f in FAULTS) == 6


def test_typed_pipeline_contains_every_value_range_fault(summary):
    stats = summary["pipeline"]["value_range"]
    assert stats["retained_flagged"] == stats["faults"] == 6
    assert stats["dropped"] == 0
    assert stats["mean_llm_calls"] == 2.0


def test_untyped_two_call_drops_every_value_range_fault_after_retries(summary):
    stats = summary["two_call"]["value_range"]
    assert stats["dropped"] == stats["faults"] == 6
    assert stats["mean_llm_calls"] == 6.0


def test_parse_level_faults_are_handled_identically_by_both_two_call_arms(summary):
    assert summary["two_call"]["parse_level"] == summary["pipeline"]["parse_level"]


def test_no_architecture_lets_an_out_of_range_value_reach_the_output(summary):
    for architecture in summary.values():
        assert architecture["all"]["out_of_range_in_output"] == 0
        assert architecture["all"]["silently_accepted"] == 0
