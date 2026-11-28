import pandas as pd

df = pd.read_csv("data/raw/paysim.csv")
print("rows before:", len(df))
df.iloc[:-1000].to_csv("data/raw/paysim.csv", index=False)
print("rows after :", len(df) - 1000)