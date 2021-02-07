import sys
from icecream import ic

import pandas as pd

sys.path.append("script")
from preprocess import conv_time_station,\
    conv_floor, floor_cnt, conv_year2int, conv_use,\
        conv_transaction, conv_square, add_tokyo_flag,\
            conv_concept, square_ratio, add_ku, conv_transaction_circumstances,\
                agg_group

from func import Feature, get_arguments, generate_features, label_encoder



class TimeToTheStation(Feature):
    def create_features(self):
        self.train['time_to_the_station'] = conv_time_station(train)["time_to_the_station"]
        self.test['time_to_the_station'] = conv_time_station(test)["time_to_the_station"]

class Floor(Feature):
    def create_features(self):  
        tmp_train = conv_floor(train)
        tmp_test = conv_floor(test)
        # tmp_train, tmp_test = label_encoder(tmp_train, tmp_test, "floor")

        col_l = ["floor_num", "floor"]
        for col in col_l:
            self.train[col] = tmp_train[col]
            self.test[col] = tmp_test[col]

class FloorCnt(Feature):
    def create_features(self):
        tmp_train = floor_cnt(train)
        tmp_test = floor_cnt(test)
        col_l = ["count_L", "count_D", "count_K",
                 "count_S", "count_R"]
        
        for col in col_l:
            self.train[col] = tmp_train[col]
            self.test[col] = tmp_test[col]

class MultiUse(Feature):
    def create_features(self):
        tmp_train = conv_use(train)
        tmp_test = conv_use(test)
        col_l = ["other", "office", "house",
                 "workplace", "warehouse",
                 "factory", "store", "parking"]
        for col in col_l:
            if col in tmp_train.columns:
                self.train[col] = tmp_train[col]
            else:
                self.train[col] = 0
            if col in tmp_test.columns:
                self.test[col] = tmp_test[col]
            else:
                self.test[col] = 0

class Year(Feature):
    def create_features(self):
        self.train['year_num'] = conv_year2int(train)["year_num"]
        self.test['year_num'] = conv_year2int(test)["year_num"]

class ConvTransaction(Feature):
    def create_features(self):
        tmp_train = conv_transaction(train)
        tmp_test = conv_transaction(test)
        col_l = ["transaction_year", "diff_year"]
        for col in col_l:
            self.train[col] = tmp_train[col]
            self.test[col] = tmp_test[col]

class Square(Feature):
    def create_features(self):
        tmp_train = conv_square(train)
        tmp_test = conv_square(test)

        col_l = ["square", "log_square",
                 "square_pref_mean", 
                 "log_square_pref_mean"]
        for col in col_l:
            self.train[col] = tmp_train[col]
            self.test[col] = tmp_test[col]

class TokyoFlag(Feature):
    def create_features(self):
        tmp_train = add_tokyo_flag(train)
        tmp_test = add_tokyo_flag(test)
        self.train['tokyo'] = tmp_train["tokyo"]
        self.test['tokyo'] = tmp_test["tokyo"]

class KuFlag(Feature):
    def create_features(self):
        tmp_train = add_ku(train)
        tmp_test = add_ku(test)
        self.train['ku_flag'] = tmp_train["ku_flag"]
        self.test['ku_flag'] = tmp_test["ku_flag"]

class ConvConcept(Feature):
    def create_features(self):
        tmp_train = conv_concept(train)
        tmp_test = conv_concept(test)
        col_l = ["RC", "SRC",
                 "iron", "wood",
                 "light_iron", "block"]
        for col in col_l:
            if col in tmp_train.columns:
                self.train[col] = tmp_train[col]
            else:
                self.train[col] = 0
            if col in tmp_test.columns:
                self.test[col] = tmp_test[col]
            else:
                self.test[col] = 0

class SquareRatio(Feature):
    def create_features(self):
        tmp_train = square_ratio(train)
        tmp_test = square_ratio(test)
        col_l = ["square*floor_area_ratio", 
                 "square*building_coverage_ratio",
                 "floor_area_by_total"]
        for col in col_l:
            self.train[col] = tmp_train[col]
            self.test[col] = tmp_test[col]

class MultiTransactionCircumstances(Feature):
    def create_features(self):
        tmp_train = conv_transaction_circumstances(train)
        tmp_test = conv_transaction_circumstances(test)
        col_l = ["Mediation_auction_etc", 
                 "transactions_between_parties",
                 "possibility_of_defect",
                 "with_other_rights_and_burdens",
                 "other_circumstances"]
        for col in col_l:
            if col in tmp_train.columns:
                self.train[col] = tmp_train[col]
            else:
                self.train[col] = 0
            if col in tmp_test.columns:
                self.test[col] = tmp_test[col]
            else:
                self.test[col] = 0

class AggGroup(Feature):
    def create_features(self):
        tmp_train = conv_square(train)
        tmp_test = conv_square(test)

        tmp_train = conv_floor(tmp_train)
        tmp_test = conv_floor(tmp_test)

        tdata = pd.concat([tmp_train, tmp_test])
        group_col = ["prefectures", "city"]
        agg_col = ["square", "building_coverage_ratio",
                   "floor_area_ratio", "floor_num",
                   "time_to_the_station", 
                   "year_num"]

        tdata = agg_group(tdata, 
                          group_col, 
                          agg_col).reset_index()
        tmp_train = tdata.loc[:tmp_train.shape[0]-1]
        tmp_test = tdata.loc[tmp_train.shape[0]:].reset_index()

        calc_col = ["mean", "std", "max", "min"]
        col_l = []

        for gcol in group_col:
            for acol in agg_col:
                for ccol in calc_col:
                    col_l.append(f"{gcol}_{acol}_{ccol}")
        for col in col_l:
            self.train[col] = tmp_train[col]
            self.test[col] = tmp_test[col]

        

class Prefectures(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "prefectures")

        self.train['prefectures'] = tmp_train["prefectures"]
        self.test['prefectures'] = tmp_test["prefectures"]

class City(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "city")

        self.train['city'] = tmp_train["city"]
        self.test['city'] = tmp_test["city"]

class District(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "district")

        self.train['district'] = tmp_train["district"]
        self.test['district'] = tmp_test["district"]

class NearestStation(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "nearest_station")

        self.train['nearest_station'] = tmp_train["nearest_station"]
        self.test['nearest_station'] = tmp_test["nearest_station"]

class Use(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "use")

        self.train['use'] = tmp_train["use"]
        self.test['use'] = tmp_test["use"]

class FuturePurposeUse(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "future_purpose_use")

        self.train['future_purpose_use'] = tmp_train["future_purpose_use"]
        self.test['future_purpose_use'] = tmp_test["future_purpose_use"]

class Concept(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "concept")

        self.train['concept'] = tmp_train["concept"]
        self.test['concept'] = tmp_test["concept"]

class CityPlanning(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "city_planning")

        self.train['city_planning'] = tmp_train["city_planning"]
        self.test['city_planning'] = tmp_test["city_planning"]

class BuildingCoverageRatio(Feature):
    def create_features(self):
        self.train['building_coverage_ratio'] = train["building_coverage_ratio"]
        self.test['building_coverage_ratio'] = test["building_coverage_ratio"]

class FloorAreaRatio(Feature):
    def create_features(self):
        self.train['floor_area_ratio'] = train["floor_area_ratio"]
        self.test['floor_area_ratio'] = test["floor_area_ratio"]

class Transaction(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "transaction")

        self.train['transaction'] = tmp_train["transaction"]
        self.test['transaction'] = tmp_test["transaction"]

class Refurbishment(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "refurbishment")

        self.train['refurbishment'] = tmp_train["refurbishment"]
        self.test['refurbishment'] = tmp_test["refurbishment"]

class TransactionCircumstances(Feature):
    def create_features(self):
        tmp_train, tmp_test = label_encoder(train, test, "transaction_circumstances")

        self.train['transaction_circumstances'] = tmp_train["transaction_circumstances"]
        self.test['transaction_circumstances'] = tmp_test["transaction_circumstances"]

class TargetEncording(Feature):
    def create_features(self):
        # レコードのtransaction以前のpriceの県ごとの平均
        # trainで集計した値をtestにも使用する
        data = pd.concat([train, test]).reset_index()
        enc_dic = {}
        for i, e in enumerate(sorted(list(set(data['transaction'].values)))):
            enc_dic[e] = i
        
        data["tra_enc"] = data["transaction"].map(enc_dic)

        te_dic = {}
        time_col = 'tra_enc'
        group_col = 'prefectures'
        TARGET = "price"

        for i in set(data[time_col].values):
            tmp_data = data[data[time_col] < i]
            te_dic[i] = tmp_data.groupby(group_col)[TARGET].agg('mean').to_dict()


        def calc_te(row):
            if row[time_col] in te_dic and row[group_col] in te_dic[row[time_col]]:
                return te_dic[row[time_col]][row[group_col]]
            else:
                return 0

        data[group_col+'_te'] = data.apply(calc_te, axis=1)

        tmp_train = data.loc[:train.shape[0]-1]
        tmp_test = data.loc[train.shape[0]:].reset_index()
        
        self.train['prefectures_te'] = tmp_train["prefectures_te"]
        self.test['prefectures_te'] = tmp_test["prefectures_te"]
    
class CountPfCity(Feature):
    def create_features(self):
        ttrain = conv_year2int(train)
        ttest = conv_year2int(test)

        data = pd.concat([ttrain, ttest]).reset_index()
        data["cnt_pf"] = data.groupby(["prefectures"]).price.transform("count")
        data["cnt_pf_city"] = data.groupby(["prefectures", "city"]).price.transform("count")

        tmp_train = data.loc[:train.shape[0]-1]
        tmp_test = data.loc[train.shape[0]:].reset_index()

        self.train['transaction_circumstances'] = tmp_train["transaction_circumstances"]
        self.test['transaction_circumstances'] = tmp_test["transaction_circumstances"]


class Price(Feature):
    def create_features(self):
        self.train['price'] = train["price"]

if __name__ == '__main__':
    Feature.dir = "features"

    args = get_arguments()

    train = pd.read_csv('output/train.csv')
    test = pd.read_csv('output/test.csv')

    generate_features(globals(), args.force)