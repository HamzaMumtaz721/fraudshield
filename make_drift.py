import pandas as pd

df = pd.read_csv("replay.csv")
money = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
late = df["step"] >= 600
df.loc[late, money] = df.loc[late, money] * 2
df.to_csv("replay_drift.csv", index=False)
print("rows changed:", int(late.sum()), "of", len(df))