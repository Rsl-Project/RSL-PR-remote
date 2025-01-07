# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

# 転移学習後モデル
MODEL_NAME = "daiki7069/t5_ad_kw_epoch8"

# データセット一時保存場所
DATA_DIR = "/home/daiki_shibata/pj/ad-generation/RSL-PR-remote/__temp__"

# データセットのURL
TEST_URL = "hf://datasets/daiki7069/camera-minimal/test.csv"
DEV_URL = "hf://datasets/daiki7069/camera-minimal/dev.csv"
TRAIN_URL = "hf://datasets/daiki7069/camera-minimal/train.csv"

# モデル一時保存場所
TEMP_MODEL_REPO = "daiki7069/temp_model"