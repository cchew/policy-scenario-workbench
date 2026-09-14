from dataclasses import dataclass, field
import pandas as pd

EVIDENCE_STRENGTHS = ("observed", "expert-assumption", "calibrated", "speculative")


@dataclass
class RegisterRow:
    parameter: str
    value: object
    source: str
    evidence_strength: str
    notes: str = ""

    def __post_init__(self):
        if self.evidence_strength not in EVIDENCE_STRENGTHS:
            raise ValueError(
                f"evidence_strength must be one of {EVIDENCE_STRENGTHS}, got {self.evidence_strength!r}"
            )


@dataclass
class Register:
    rows: list = field(default_factory=list)

    def add(self, row: RegisterRow) -> None:
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"parameter": r.parameter, "value": r.value, "source": r.source,
             "evidence_strength": r.evidence_strength, "notes": r.notes}
            for r in self.rows
        ])

    def write_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def assert_covers(self, param_ids: set) -> None:
        present = {r.parameter for r in self.rows}
        missing = param_ids - present
        assert not missing, f"Register missing required parameters: {sorted(missing)}"
