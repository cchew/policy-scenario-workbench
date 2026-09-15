from pathlib import Path

import pandas as pd
import pytest

from src import schema

RAW_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ai_adoption_vietnam.csv"

_real_csv_missing = pytest.mark.skipif(
    not RAW_CSV_PATH.exists(),
    reason="real dataset not present locally — see data/raw/README.md",
)


def test_construct_item_columns_cover_all_five_constructs():
    assert set(schema.RAW_TO_ITEM_COLUMNS.keys()) == {"pe", "ee", "si", "fc", "bi"}
    for construct, cols in schema.RAW_TO_ITEM_COLUMNS.items():
        assert len(cols) >= 1, f"{construct} has no raw item columns filled in"


@_real_csv_missing
def test_construct_item_columns_exist_in_real_csv():
    df = pd.read_csv(RAW_CSV_PATH)
    for construct, cols in schema.RAW_TO_ITEM_COLUMNS.items():
        for col in cols:
            assert col in df.columns, (
                f"{construct} raw column not found in real CSV header: {col!r}"
            )


def test_demographic_columns_present():
    assert set(schema.RAW_DEMOGRAPHIC_COLUMNS.keys()) == {"role"}
    for key, raw_name in schema.RAW_DEMOGRAPHIC_COLUMNS.items():
        assert raw_name, f"{key} has no raw column name filled in"


@_real_csv_missing
def test_demographic_columns_exist_in_real_csv():
    df = pd.read_csv(RAW_CSV_PATH)
    for raw_name in schema.RAW_DEMOGRAPHIC_COLUMNS.values():
        assert raw_name in df.columns


def test_likert_max_is_five():
    assert schema.LIKERT_MAX == 5


def test_normalized_columns_match_expected_set():
    assert schema.NORMALIZED_COLUMNS == [
        "respondent_id", "role",
        "pe_score", "ee_score", "si_score", "fc_score", "behavioral_intention",
    ]


def test_reverse_coded_constructs():
    assert schema.REVERSE_CODED_CONSTRUCTS == {"fc"}


def test_role_value_map():
    assert schema.ROLE_VALUE_MAP == {
        "Giảng viên": "staff",
        "Sinh viên": "student",
    }


def test_parse_likert_value():
    assert schema.parse_likert_value("4 (Đồng ý)") == 4
    assert schema.parse_likert_value("1 (Hoàn toàn không đồng ý)") == 1
