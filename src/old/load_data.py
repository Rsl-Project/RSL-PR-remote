import pandas as pd
import os

from env import DATA_DIR

# データセット保存用の一時ディレクトリを作成
os.makedirs(DATA_DIR)

df_test = pd.read_csv("hf://datasets/daiki7069/camera-minimal/test.csv")
df_dev = pd.read_csv("hf://datasets/daiki7069/camera-minimal/dev.csv")
df_train = pd.read_csv("hf://datasets/daiki7069/camera-minimal/train.csv")
