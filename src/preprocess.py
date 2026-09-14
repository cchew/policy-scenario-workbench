"""Clean the raw Mendeley AI-adoption survey export into the normalized
schema defined in src/schema.py, score each construct, and dichotomize
behavioral intention into a top-2-box `adopter` label.
"""
import pandas as pd

from src import schema


def _construct_score(df: pd.DataFrame, construct: str) -> pd.Series:
    cols = schema.RAW_TO_ITEM_COLUMNS[construct]
    parsed = df[cols].apply(lambda c: c.map(schema.parse_likert_value))
    if construct in schema.REVERSE_CODED_CONSTRUCTS:
        parsed = (schema.LIKERT_MAX + 1) - parsed
    return parsed.mean(axis=1)


def load_and_clean(raw_csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv_path)
    raw = raw.dropna(subset=[
        c for cols in schema.RAW_TO_ITEM_COLUMNS.values() for c in cols
    ])

    out = pd.DataFrame()
    out["respondent_id"] = raw.index
    out["role"] = raw[schema.RAW_DEMOGRAPHIC_COLUMNS["role"]].map(schema.ROLE_VALUE_MAP)
    out["pe_score"] = _construct_score(raw, "pe")
    out["ee_score"] = _construct_score(raw, "ee")
    out["si_score"] = _construct_score(raw, "si")
    out["fc_score"] = _construct_score(raw, "fc")
    out["behavioral_intention"] = _construct_score(raw, "bi")

    top_2_box_threshold = schema.LIKERT_MAX - 1
    out["adopter"] = (out["behavioral_intention"] >= top_2_box_threshold).astype(int)

    return out.reset_index(drop=True)


def check_class_balance(df: pd.DataFrame, min_frac: float = 0.1) -> None:
    frac_adopters = df["adopter"].mean()
    if frac_adopters < min_frac or frac_adopters > (1 - min_frac):
        raise ValueError(
            f"Degenerate adopter class split: {frac_adopters:.1%} adopters. "
            f"Top-2-box threshold may need revisiting (see Known limitations "
            f"in the design spec)."
        )
