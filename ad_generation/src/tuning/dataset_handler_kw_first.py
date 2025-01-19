import pandas as pd
import os
from src.util.env import DATA_DIR
from src.util.normalization import normalize_text

class DatasetHandlerKwFirst():
    def __init__(self, test_url, dev_url, train_url):
        # self.test_url = test_url
        self.dev_url = dev_url
        self.train_url = train_url


    def _load_data(self):
        '''
            データセット保存用の一時ディレクトリを作成.
        '''
        if not(os.path.isdir(DATA_DIR)):
            os.makedirs(DATA_DIR)
        # if not(os.path.isfile("test.tsv")):
        #     self.df_test = pd.read_csv(self.test_url)
        if not(os.path.isfile("dev.tsv")):
            self.df_dev = pd.read_csv(self.dev_url)
        if not(os.path.isfile("train.tsv")):
            self.df_train = pd.read_csv(self.train_url)
        
        
    def csv2tsv(self) -> None:
        '''
            csvファイルからtsvファイルに変換.
        '''
        self._load_data()
        df_train_kw = pd.read_csv("hf://datasets/daiki7069/camera-minimal/train.csv")
        df_dev_kw = pd.read_csv("hf://datasets/daiki7069/camera-minimal/dev.csv")

        top_tfidf_words = [
            "価格",
            "無料",
            "すぐ",
            "徒歩",
            "保険",
            "限定",
            "公式",
            "2022",
            "公式",
        ]
        print(top_tfidf_words)

        # 学習データ
        self.df_train['kw'] = df_train_kw['kw']
        self.df_train = self.df_train.drop(columns=self.df_train.columns[0])
        self.df_train = self.df_train.drop(columns=[
            "asset_id",
            "lp_meta_description",
        ])
        self.df_train["title_org"] = self.df_train["title_org"].astype(str)
        self.df_train["title_org"] = self.df_train["title_org"].map(lambda s: s.replace(" ", "/"))
        self.df_train = self.df_train.map(normalize_text)
        self.df_train = self.df_train.dropna()
        self.df_train["kw"] = self.df_train.apply(
            # lambda row: f"[aa{int(row['appealing_axis'])}]{row['kw']}" if row['appealing_axis'] in [0, 1, 2, 3, 4, 5, 6, 7, 8] else row['kw'],
            lambda row: f"{top_tfidf_words[int(row['appealing_axis'])]}{row['kw']}" if row['appealing_axis'] in [0, 1, 2, 3, 4, 5, 6, 7, 8] else row['kw'],
            axis=1
        )
        self.df_train = self.df_train.drop(columns=["appealing_axis"])
        self.df_train.to_csv(f"{DATA_DIR}/train.tsv", sep="\t", index=False, header=None, encoding="utf-8")


        # 学習中の精度評価に使用するdevデータ
        self.df_dev['kw'] = df_dev_kw['kw']
        self.df_dev = self.df_dev.drop(columns=self.df_dev.columns[0])
        self.df_dev = self.df_dev.drop(columns=[
            "asset_id",
            "lp_meta_description",
        ])
        self.df_dev["title_org"] = self.df_dev["title_org"].astype(str)
        self.df_dev["title_org"] = self.df_dev["title_org"].map(lambda s: s.replace(" ", "/"))
        self.df_dev = self.df_dev.map(normalize_text)
        self.df_dev = self.df_dev.dropna()
        self.df_dev["kw"] = self.df_dev.apply(
            # lambda row: f"[aa{int(row['appealing_axis'])}]{row['kw']}" if row['appealing_axis'] in [0, 1, 2, 3, 4, 5, 6, 7, 8] else row['kw'],
            lambda row: f"{top_tfidf_words[int(row['appealing_axis'])]}{row['kw']}" if row['appealing_axis'] in [0, 1, 2, 3, 4, 5, 6, 7, 8] else row['kw'],
            axis=1
        )
        self.df_dev = self.df_dev.drop(columns=["appealing_axis"])
        self.df_dev.to_csv(f"{DATA_DIR}/dev.tsv", sep="\t", index=False, header=None, encoding="utf-8")



    def remove_tsv(self) -> None:
        '''
            tsvファイルを削除.
        '''
        os.remove(f"{DATA_DIR}/dev.tsv")
        # os.remove(f"{DATA_DIR}/test.tsv")
        os.remove(f"{DATA_DIR}/train.tsv")
        os.rmdir(DATA_DIR)

