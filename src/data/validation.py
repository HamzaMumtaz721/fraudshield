import pandas as pd

REQUIRED = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"]
NUMERIC = [c for c in REQUIRED if c != "type"]
MONEY = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
VALID_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}


class DataValidationError(ValueError):
    pass


def validation_problems(df: pd.DataFrame) -> list:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    problems = [f"{c} is not numeric" for c in NUMERIC if not pd.api.types.is_numeric_dtype(df[c])]
    if problems:
        return problems
    for c, n in df[REQUIRED].isnull().sum().items():
        if n:
            problems.append(f"{c} has {int(n)} missing values")
    bad_types = set(df["type"].dropna().unique()) - VALID_TYPES
    if bad_types:
        problems.append(f"unknown transaction types: {sorted(bad_types)}")
    if (df["step"] < 1).any():
        problems.append("step has values below 1")
    for c in MONEY:
        if (df[c] < 0).any():
            problems.append(f"{c} has negative values")
    if not set(df["isFraud"].unique()) <= {0, 1}:
        problems.append("isFraud has values other than 0 and 1")
    return problems


def validate(df: pd.DataFrame) -> None:
    problems = validation_problems(df)
    if problems:
        raise DataValidationError("; ".join(problems))