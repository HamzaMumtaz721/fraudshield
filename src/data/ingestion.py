import hashlib

import pandas as pd

USECOLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest", "isFraud"]


def load_raw(path: str) -> pd.DataFrame:
    """Load the PaySim CSV. Account-name columns are skipped: the model never uses them."""
    return pd.read_csv(path, usecols=USECOLS)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()