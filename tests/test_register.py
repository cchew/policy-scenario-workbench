import pytest
from src.register import Register, RegisterRow, EVIDENCE_STRENGTHS


def test_register_row_rejects_invalid_evidence_strength():
    with pytest.raises(ValueError):
        RegisterRow(parameter="x", value=1, source="s", evidence_strength="guess", notes="")


def test_register_add_and_to_dataframe():
    reg = Register()
    reg.add(RegisterRow("dcm.coef.pe", 0.42, "survey fit", "observed", "test"))
    df = reg.to_dataframe()
    assert list(df.columns) == ["parameter", "value", "source", "evidence_strength", "notes"]
    assert df.iloc[0]["parameter"] == "dcm.coef.pe"


def test_assert_covers_raises_on_missing_ids():
    reg = Register()
    reg.add(RegisterRow("dcm.coef.pe", 0.42, "survey fit", "observed", ""))
    with pytest.raises(ValueError):
        reg.assert_covers({"dcm.coef.pe", "dcm.coef.ee"})


def test_assert_covers_passes_when_all_present():
    reg = Register()
    reg.add(RegisterRow("dcm.coef.pe", 0.42, "survey fit", "observed", ""))
    reg.add(RegisterRow("dcm.coef.ee", 0.10, "survey fit", "observed", ""))
    reg.assert_covers({"dcm.coef.pe", "dcm.coef.ee"})  # should not raise
