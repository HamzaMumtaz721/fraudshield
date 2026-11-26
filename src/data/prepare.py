import json
from pathlib import Path

import yaml

from src.data.ingestion import load_raw
from src.data.validation import validate
from src.features.preprocessing import add_features


def main():
    cfg = yaml.safe_load(open("configs/model_params.yaml"))
    d = cfg["data"]
    df = load_raw(d["raw_path"])
    validate(df)                       # raises, and the stage fails, if the data is bad
    df = add_features(df[df["type"].isin(d["types_in_scope"])])

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "transactions.parquet", index=False)

    report = {"validation": "passed", "rows_in_scope": int(len(df)),
              "frauds_in_scope": int(df["isFraud"].sum())}
    for key in ("train_steps", "val_steps", "test_steps"):
        lo, hi = d[key]
        part = df[(df["step"] >= lo) & (df["step"] <= hi)]
        report[key] = {"rows": int(len(part)), "frauds": int(part["isFraud"].sum())}
    json.dump(report, open(out / "prepare_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()