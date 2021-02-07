import json
import os
import pandas as pd


INPUT = "input/"
CONFIG = "config/"
OUTPUT = "output/"

with open(f"{CONFIG}/col_en.json", "r") as f:
    col_dic = json.load(f)

train = pd.DataFrame()
for i, file in enumerate(os.listdir(INPUT+"/train")):
    df = pd.read_csv(INPUT + f"train/{file}")
    train = pd.concat([train, df], axis=0)

test = pd.read_csv(INPUT + "test.csv")

train = train.rename(columns=col_dic)
test = test.rename(columns=col_dic)

train.to_csv(OUTPUT + "train.csv", index=False)
test.to_csv(OUTPUT + "test.csv", index=False)
