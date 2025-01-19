import pandas as pd
from math import isnan

df = pd.read_csv("hf://datasets/daiki7069/auto_classification/train.csv")

df['appealing_axis'] = df['appealing_axis'].fillna(8)

df.to_csv("./train.csv")