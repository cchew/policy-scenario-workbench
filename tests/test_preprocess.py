import pandas as pd
import pytest

from src import preprocess


def test_load_and_clean_produces_normalized_columns():
    df = preprocess.load_and_clean("tests/fixtures/sample_raw.csv")
    for col in ["respondent_id", "role", "pe_score", "ee_score",
                "si_score", "fc_score", "behavioral_intention", "adopter"]:
        assert col in df.columns

    assert "faculty" not in df.columns
    assert df["adopter"].isin([0, 1]).all()
    assert len(df) == 10


def test_role_is_mapped_to_normalized_values():
    df = preprocess.load_and_clean("tests/fixtures/sample_raw.csv")
    assert set(df["role"].unique()) <= {"staff", "student"}


def test_adopter_is_top_2_box_of_behavioral_intention():
    df = preprocess.load_and_clean("tests/fixtures/sample_raw.csv")
    top_2_box_threshold = preprocess.schema.LIKERT_MAX - 1
    for _, row in df.iterrows():
        expected = int(row["behavioral_intention"] >= top_2_box_threshold)
        assert row["adopter"] == expected


def test_check_class_balance_raises_on_degenerate_split():
    all_adopters = pd.DataFrame({"adopter": [1] * 20})
    with pytest.raises(ValueError):
        preprocess.check_class_balance(all_adopters, min_frac=0.1)


def test_check_class_balance_passes_on_reasonable_split():
    mixed = pd.DataFrame({"adopter": [1] * 6 + [0] * 4})
    preprocess.check_class_balance(mixed, min_frac=0.1)  # should not raise


def test_fc_score_is_reverse_coded(tmp_path):
    """fc raw items are worded as concerns/barriers: a row with all raw
    items at '1' (low concern) should produce a HIGHER fc_score than a row
    with all raw items at '5' (high concern), since fc is in
    schema.REVERSE_CODED_CONSTRUCTS.
    """
    from src import schema

    role_col = schema.RAW_DEMOGRAPHIC_COLUMNS["role"]
    all_cols = [c for cols in schema.RAW_TO_ITEM_COLUMNS.values() for c in cols]
    fc_cols = schema.RAW_TO_ITEM_COLUMNS["fc"]

    def make_row(fc_label):
        row = {role_col: "Sinh viên"}
        for col in all_cols:
            row[col] = fc_label if col in fc_cols else "3 (Trung lập)"
        return row

    low_concern_row = make_row("1 (Hoàn toàn không đồng ý)")  # raw 1s -> high fc_score after reverse
    high_concern_row = make_row("5 (Hoàn toàn đồng ý)")  # raw 5s -> low fc_score after reverse

    df = pd.DataFrame.from_records([low_concern_row, high_concern_row])
    csv_path = tmp_path / "fc_reverse_fixture.csv"
    df.to_csv(csv_path, index=False)

    out = preprocess.load_and_clean(str(csv_path))
    assert out.loc[0, "fc_score"] > out.loc[1, "fc_score"]
