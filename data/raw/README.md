# Raw data

Source: Mendeley Data, `pyyjfthc84` —
https://data.mendeley.com/datasets/pyyjfthc84/1

The Mendeley download is `UTAUT_AI_in_University.xlsx` (a Google Forms export),
**not** a CSV. To reproduce `ai_adoption_vietnam.csv` in this directory:

1. Download the `.xlsx` from the URL above.
2. Convert the `Form Responses 1` sheet to CSV losslessly (no value changes),
   e.g. `pd.read_excel("UTAUT_AI_in_University.xlsx", sheet_name="Form Responses 1").to_csv("ai_adoption_vietnam.csv", index=False)`.
   Requires `openpyxl` to read the `.xlsx` (not a project dependency — only
   needed for this one-off conversion step).
3. Save the result here as `ai_adoption_vietnam.csv`.

This directory is gitignored — the file is not committed to either repo. If
you don't want to redo the conversion yourself, ask whoever has a working copy
of this repo for the already-converted CSV.

Shape: 59 rows x 42 columns. See `docs/dataset-schema-notes.md` (parent repo,
`projects/policy-scenario-workbench/docs/`) for the column mapping and
discovery notes, and `src/schema.py` for the normalized schema consumed by the
rest of this codebase.
