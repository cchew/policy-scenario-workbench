import json


def evaluate_rule(rule: dict, facts: dict) -> bool:
    op, args = next(iter(rule.items()))

    def resolve(arg):
        if isinstance(arg, dict) and "var" in arg:
            return facts[arg["var"]]
        return arg

    if op == "var":
        return facts[args]
    if op == "in":
        needle, haystack = resolve(args[0]), resolve(args[1])
        return needle in haystack
    if op == "and":
        return all(evaluate_rule(sub, facts) for sub in args)
    if op == "or":
        return any(evaluate_rule(sub, facts) for sub in args)
    if op == "==":
        return resolve(args[0]) == resolve(args[1])
    if op == "!=":
        return resolve(args[0]) != resolve(args[1])
    raise ValueError(f"Unsupported operator: {op}")


def load_eligibility_rule(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_eligibility(df, rule_doc: dict):
    df = df.copy()
    df["eligible"] = df.apply(
        lambda row: evaluate_rule(rule_doc["rule"], row.to_dict()), axis=1
    )
    return df
