import pandas as pd
import os
from src.util.env import DATA_DIR
from src.util.normalization import normalize_text

class DatasetHandlerKw():
    def __init__(self, test_url, dev_url, train_url):
        self.test_url = test_url
        self.dev_url = dev_url
        self.train_url = train_url


    def _load_data(self):
        '''
            データセット保存用の一時ディレクトリを作成.
        '''
        if not(os.path.isdir(DATA_DIR)):
            os.makedirs(DATA_DIR)
        if not(os.path.isfile("test.tsv")):
            self.df_test = pd.read_csv(self.test_url)
        if not(os.path.isfile("dev.tsv")):
            self.df_dev = pd.read_csv(self.dev_url)
        if not(os.path.isfile("train.tsv")):
            self.df_train = pd.read_csv(self.train_url)
        
        
    def csv2tsv(self) -> None:
        '''
            csvファイルからtsvファイルに変換.
        '''
        self._load_data()

        # 学習データ
        self.df_train = self.df_train.drop(columns=[
            "lp_meta_description",
            "parsed_full_text_annotation",
        ])
        self.df_train["kw"] = self.df_train["kw"].map(lambda s: s.replace(" ", "/"))
        self.df_train = self.df_train.map(normalize_text)
        self.df_train = self.df_train.dropna()
        self.df_train.to_csv(f"{DATA_DIR}/train.tsv", sep="\t", index=False, header=None, encoding="utf-8")

        # 学習中の精度評価に使用するdevデータ
        self.df_dev = self.df_dev.drop(columns=[
            "lp_meta_description",
            "parsed_full_text_annotation",
        ])
        self.df_dev["kw"] = self.df_dev["kw"].map(lambda s: s.replace(" ", "/"))
        self.df_dev = self.df_dev.map(normalize_text)
        self.df_dev = self.df_dev.dropna()
        self.df_dev.to_csv(f"{DATA_DIR}/dev.tsv", sep="\t", index=False, header=None, encoding="utf-8")

        # テストデータ
        self.df_test = self.df_test.drop(columns=[
            "lp_meta_description",
            "title_ne1",
            "title_ne2",
            "title_ne3",
            "domain",
            "parsed_full_text_annotation",
        ])
        self.df_test["kw"] = self.df_test["kw"].map(lambda s: s.replace(" ", "/"))
        self.df_test = self.df_test.map(normalize_text)
        self.df_test = self.df_test.dropna()
        self.df_test.to_csv(f"{DATA_DIR}/test.tsv", sep="\t", index=False, header=None, encoding="utf-8")


    def remove_tsv(self) -> None:
        '''
            tsvファイルを削除.
        '''
        os.remove(f"{DATA_DIR}/dev.tsv")
        os.remove(f"{DATA_DIR}/test.tsv")
        os.remove(f"{DATA_DIR}/train.tsv")
        os.rmdir(DATA_DIR)

