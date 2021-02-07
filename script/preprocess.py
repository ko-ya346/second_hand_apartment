import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def conv_data(data):
    data = conv_time_station(data)
    data = conv_floor(data)
    data = floor_cnt(data)
    data = conv_square(data)
    data.year = data.year.fillna("nan")
    data.year = data.year.apply(lambda x: conv_year(x))
    data = conv_year2int(data).reset_index(drop=True)
    
    year_num_dic = \
        data.groupby("prefectures")["year_num"].mean().to_dict()
    null_yn = data.year_num.isnull()

    data.loc[null_yn, 
             "year_num"] = data.loc[null_yn, 
                                    "prefectures"].map(year_num_dic)
    data.year_num = data.year_num.astype(int)
    data = conv_use(data)
    data = conv_transaction(data)
    # data = fill_time_station(data)
    data = add_tokyo_flag(data)
    data = add_ku(data)
    data = conv_concept(data)
    data = conv_transaction_circumstances(data)
    data = square_ratio(data)
    return data

def conv_time_station(data):
    # 最寄り駅までの時間を数値に変換
    time_to_the_station_dic = {"30分?60分": 45, 
                               '1H?1H30': 75, 
                               "2H?": 120,
                               '1H30?2H': 105}
    data.loc[data.time_to_the_station.isin(
        time_to_the_station_dic.keys()
        ), 
        "time_to_the_station"] = \
            data.loc[data.time_to_the_station.isin(
                time_to_the_station_dic.keys()
                ), 
                "time_to_the_station"].map(
                    time_to_the_station_dic
                    )
    data.time_to_the_station = data.time_to_the_station.astype(float)
    return data

def conv_floor(data):
    # floorを変換、部屋数を抽出
    replace_l = [
        ("１", "1"), ("２", "2"), ("３", "3"), 
        ("４", "4"), ("５", "5"), ("６", "6"), ("７", "7"), 
        ("８", "8"), ("Ｄ", "D"), ("Ｋ", "K"), ("＋", "+"), 
        ("Ｓ", "S"), ("Ｌ", "L"), ("Ｒ", "R"), 
        ('オープンフロア', "open_floor"), 
        ('スタジオ', "studio"), ('メゾネット', "maisonette")
        ]

    for before, after in replace_l:
        data.floor = data.floor.str.replace(before, after)
    
    data["floor_num"] = data.floor.str.extract("(\d)")
    data["floor_num"] = data["floor_num"].fillna(0).astype(int)
    return data

def floor_cnt(data):
    data.floor = data.floor.fillna("nan")
    data["count_L"] = data.floor.apply(lambda x: x.count("L"))
    data["count_D"] = data.floor.apply(lambda x: x.count("D"))
    data["count_K"] = data.floor.apply(lambda x: x.count("K"))
    data["count_S"] = data.floor.apply(lambda x: x.count("S"))
    data["count_R"] = data.floor.apply(lambda x: x.count("R"))
    data.loc[data.floor=="nan", "floor"] = np.nan
    return data

def conv_square(data):
    # square
    data.loc[data.square=='2000㎡以上', 
             "square"] = "2000"
    data.square = data.square.astype(float)
    data["log_square"] = np.log1p(data.square)
    
    pref_square_dic = data.groupby("prefectures").square.mean().to_dict()
    data["square_pref_mean"] = data["prefectures"].map(pref_square_dic)
    data["log_square_pref_mean"] = np.log1p(data["square_pref_mean"])
    return data

def conv_year(word):
    # year
    if "元" in word:
        word = word.replace("元", "1")
    if word == "戦前":
        word = "昭和20年"
    return word

def conv_year2int(data):
    data["heisei"] = data.year.str.contains("平成")
    data["syouwa"] = data.year.str.contains("昭和")
    data["reiwa"] = data.year.str.contains("令和")
    data["year_num"] = data.year.str.extract("(\d)").\
        astype(float)
    data.loc[data.heisei==True, "year_num"] += 1988
    data.loc[data.syouwa==True, "year_num"] += 1925
    data.loc[data.reiwa==True, "year_num"] += 2018

    data["before_year"] = 2020 - data.year_num
    return data

def conv_use(data):
    arr_dic = {'none': 'none', "その他": 'other', 
    "事務所": 'office', "住宅": 'house', 
    "作業場": 'workplace', "倉庫": 'warehouse',
    "工場": 'factory', "店舗": 'store', "駐車場": 'parking'}

    use = data.use.fillna("none").str.split("、")
    mlb = MultiLabelBinarizer() 
    arr = mlb.fit_transform(use) 
    use_l = mlb.classes_

    for i in range(len(use_l)):
        use_l[i] = arr_dic[use_l[i]]

    use_df = pd.DataFrame(arr, columns=use_l)
    data = pd.concat([data, use_df], axis=1)
    return data

def conv_transaction(data):
    data["transaction_year"] = \
        data.transaction.str.extract("(\d+)").astype(int)
    data["shihanki"] = \
        data["transaction"].apply(lambda x: transaction_shihanki(x))
    data["diff_year"] = (data["transaction_year"] 
                       + data["shihanki"] 
                       - data["year_num"])
    data.loc[data.diff_year < 0, "diff_year"] = 0
    return data

def transaction_shihanki(word):
    ret = 0
    if "１" in word:
        ret = 0.25
    elif "２" in word:
        ret = 0.5
    elif "３" in word:
        ret = 0.75
    else:
        ret = 1
    return ret
    
def fill_time_station(data):
    # districtの欠損値を、最寄り駅の最頻値で埋める
    for nearest in data[data.district.isnull()].nearest_station.dropna().unique():
        is_nearest = data.nearest_station==nearest
        dic = data[is_nearest].district.value_counts().to_dict()
        max_district = sorted(dic.items(), 
                              key=lambda x: x[1], reverse=True)[0][0]
        data.loc[is_nearest, "district"] = max_district
    
    # 最寄り駅の欠損値を、districtの最頻値で埋める
    for district in data[data.nearest_station.isnull()].district.dropna().unique():
        is_district = data.district==district
        dic = data[is_district].nearest_station.value_counts().to_dict()
        if len(dic)==0:
            continue
        max_nearest_station = sorted(dic.items(), 
                                     key=lambda x: x[1], reverse=True)[0][0]
        data.loc[is_district, "nearest_station"] = max_nearest_station

    time_fill_col = ["nearest_station", "district", 
                     "city", "prefectures"]
    data.time_to_the_station = \
        data.time_to_the_station.astype(float)

    # 最寄り駅までの時間を付近の情報の平均値で埋める
    for col in time_fill_col:
        mean_station_time_dic = \
            data.groupby(col).time_to_the_station.mean().to_dict()
        isnull_time_to_the_station = \
            data.time_to_the_station.isnull()
        data.loc[isnull_time_to_the_station, 
                 "time_to_the_station"] \
                     = data.loc[isnull_time_to_the_station, 
                                col].map(mean_station_time_dic)
        data.time_to_the_station = \
            data.time_to_the_station.astype(float)
    return data

def add_tokyo_flag(data):
    data["tokyo"] = data["prefectures"]=="東京都"
    return data

def add_ku(data):
    data["ku_flag"] = data["city"].str.contains("区")
    return data

def conv_concept(data):
    arr_dic = {'none': 'none2', "ＲＣ": 'RC', 
    "ＳＲＣ": 'SRC', "鉄骨造": 'iron', 
    "木造": 'wood', "軽量鉄骨造": 'light_iron',
    "ブロック造": 'block'}

    concept = data.concept.fillna("none").str.split("、")
    mlb = MultiLabelBinarizer() 
    arr = mlb.fit_transform(concept) 
    concept_l = mlb.classes_

    for i in range(len(concept_l)):
        concept_l[i] = arr_dic[concept_l[i]]

    concept_df = pd.DataFrame(arr, columns=concept_l)
    data = pd.concat([data, concept_df], axis=1)
    return data

def square_ratio(data):
    data["square*floor_area_ratio"] = \
        data.floor_area_ratio * data.square
    data["square*building_coverage_ratio"] = \
        data.building_coverage_ratio * data.square
    data["floor_area_by_total"] = data["square*floor_area_ratio"] / data["square*building_coverage_ratio"]
    return data

def conv_transaction_circumstances(data):
    arr_dic = {
        'none': 'none3', 
        "調停・競売等": 'Mediation_auction_etc', 
        "関係者間取引": 'transactions_between_parties', 
        "瑕疵有りの可能性": 'possibility_of_defect', 
        "他の権利・負担付き": 'with_other_rights_and_burdens',
        "その他事情有り": "other_circumstances"
               }

    tc = data.transaction_circumstances.fillna("none").str.split("、")
    mlb = MultiLabelBinarizer() 
    arr = mlb.fit_transform(tc) 
    tc_l = mlb.classes_

    for i in range(len(tc_l)):
        tc_l[i] = arr_dic[tc_l[i]]

    tc_df = pd.DataFrame(arr, columns=tc_l)
    data = pd.concat([data, tc_df], axis=1)
    return data

def agg_group(data, group_col, agg_col):
    for gcol in group_col:
        for acol in agg_col:
            tdf = data.groupby(gcol)[acol].agg({"mean", "std", "max", "min"}).reset_index()
            agg_l = ["mean", "std", "max", "min"]
            col_name = [f"{gcol}_{acol}_{agg}" for agg in agg_l]
            col_dic = dict(zip(agg_l, col_name))
            tdf = tdf.rename(columns=col_dic)
            data = pd.merge(data, tdf, on=gcol, how="left")
    return data



if __name__ == "__main__":
    INPUT = "input/"
    OUTPUT = "output/"

    train = pd.read_csv(OUTPUT + "train.csv")
    test = pd.read_csv(OUTPUT + "test.csv")

    data = pd.concat([train, test])

    data = conv_data(data)

    del_col = ["type", "year", "city_code", "area", 
            "land_shape", "frontage", "total_floor_area", 
            "FR_direction", "FR_type", "FR_width",
            "heisei", "reiwa", "syouwa", "use",
            "transaction", "none", 
            "none2", "none3",
            "transaction_circumstances"]
    data.drop(del_col, 
            axis=1, inplace=True)

    train = data.iloc[:train.shape[0]]
    test = data.iloc[train.shape[0]:]

    print(train.shape)
    print(test.shape)

    train.to_csv(OUTPUT + "preprocess_train.csv",
                index=False)
    test.to_csv(OUTPUT + "preprocess_test.csv",
                index=False)