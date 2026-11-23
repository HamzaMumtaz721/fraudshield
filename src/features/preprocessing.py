import pandas as pd


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add the model's derived columns. Input needs: step, type, amount and the four balances."""
    d = d.copy()
    d["hour"] = d["step"] % 24
    d["is_transfer"] = (d["type"] == "TRANSFER").astype(int)
    d["errorOrig"] = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]
    d["errorDest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
    return d