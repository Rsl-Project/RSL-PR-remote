import pandas as pd

# 学習データ
df_train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", delimiter=",", encoding="utf-8")
df_train = df_train_raw.drop(columns=[
    "parsed_full_text_annotation",
])
df_train = df_train.map(normalize_text) # TODO: kwのみに空白->/処理を適用
# df_train = df_train["kw"].map(lambda s: s.replace(" ", "/"))
df_train = df_train.dropna()
df_train.to_csv(f"{DATA_DIR}/train.tsv", sep="\t", index=False, header=None, encoding="utf-8")

# 学習中の精度評価に使用するdevデータ
df_dev_raw = pd.read_csv(f"{DATA_DIR}/dev.csv", delimiter=",", encoding="utf-8")
df_dev = df_dev_raw.drop(columns=[
    "parsed_full_text_annotation",
])
df_dev = df_dev.map(normalize_text)   # TODO: kwのみに空白->/処理を適用
df_dev = df_dev.dropna()
df_dev.to_csv(f"{DATA_DIR}/dev.tsv", sep="\t", index=False, header=None, encoding="utf-8")

# テストデータ
df_test_raw = pd.read_csv(f"{DATA_DIR}/test.csv", delimiter=",", encoding="utf-8")
df_test = df_test_raw.drop(columns=[
    "title_ne1",
    "title_ne2",
    "title_ne3",
    "domain",
    "parsed_full_text_annotation",
])
df_test = df_test.map(normalize_text)  # TODO: kwのみに空白->/処理を適用
df_test = df_test.dropna()
df_test.to_csv(f"{DATA_DIR}/test.tsv", sep="\t", index=False, header=None, encoding="utf-8")