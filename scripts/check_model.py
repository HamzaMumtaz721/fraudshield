import json
import xgboost as xgb

meta = json.load(open("artifacts/meta.json"))
print("features:", meta["features"])
print("threshold:", meta["threshold"])

model = xgb.XGBClassifier()
model.load_model("artifacts/xgb_full.json")
print("model loaded OK")