import pandas as pd
import pytest

from src import rac


def test_evaluate_rule_equality_operator_true():
    rule = {"==": [{"var": "role"}, "staff"]}
    assert rac.evaluate_rule(rule, {"role": "staff"}) is True


def test_evaluate_rule_equality_operator_false():
    rule = {"==": [{"var": "role"}, "staff"]}
    assert rac.evaluate_rule(rule, {"role": "student"}) is False


def test_evaluate_rule_missing_fact_key_raises():
    rule = {"==": [{"var": "role"}, "staff"]}
    with pytest.raises(KeyError):
        rac.evaluate_rule(rule, {})


def test_load_eligibility_rule_has_rule_and_note():
    doc = rac.load_eligibility_rule("rules/eligibility.json")
    assert "rule" in doc
    assert "note" in doc
    assert doc["note"]  # non-empty


def test_apply_eligibility_adds_boolean_column():
    df = pd.DataFrame({"role": ["staff", "student"]})
    doc = rac.load_eligibility_rule("rules/eligibility.json")
    out = rac.apply_eligibility(df, doc)
    assert out["eligible"].tolist() == [True, False]
