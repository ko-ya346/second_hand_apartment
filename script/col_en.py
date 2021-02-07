import json
import pandas as pd


INPUT = "input/train/"
OUTPUT = "config/"
train1 = pd.read_csv(INPUT + "01.csv")

col_en = [
    "id", "type", "area", "city_code", 
    "prefectures", "city", "district",
    "nearest_station", 
    "time_to_the_station", 
    "floor", "square", "land_shape",
    "frontage", "total_floor_area",
    "year", "concept", "use", 
    "future_purpose_use", 
    "FR_direction", "FR_type", "FR_width", 
    "city_planning", 
    "building_coverage_ratio", 
    "floor_area_ratio", 
    "transaction", "refurbishment", 
    "transaction_circumstances", "price"
    ]

col = train1.columns
col_dic = dict(zip(col, col_en))

with open(f"{OUTPUT}/col_en.json", "w") as f:
    json.dump(col_dic, f, indent=4)