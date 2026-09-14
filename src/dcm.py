import statsmodels.api as sm
import pandas as pd

from src.register import RegisterRow

FEATURES = ["pe_score", "ee_score", "si_score", "fc_score"]
_SHORT_NAMES = {"pe_score": "pe", "ee_score": "ee", "si_score": "si", "fc_score": "fc"}


def fit_dcm(df: pd.DataFrame):
    X = sm.add_constant(df[FEATURES])
    y = df["adopter"]
    return sm.Logit(y, X).fit(disp=0)


def dcm_register_rows(result) -> list[RegisterRow]:
    rows = []
    for name, coef in result.params.items():
        if name == "const":
            param_id = "dcm.intercept"
        else:
            param_id = f"dcm.coef.{_SHORT_NAMES[name]}"
        rows.append(RegisterRow(
            parameter=param_id,
            value=float(coef),
            source="Mendeley UTAUT survey (pyyjfthc84), statsmodels Logit fit",
            evidence_strength="observed",
            notes=f"p={result.pvalues[name]:.4f}",
        ))
    return rows


def predict_probability(result, df: pd.DataFrame) -> pd.Series:
    X = sm.add_constant(df[FEATURES], has_constant="add")
    return result.predict(X)
