# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"
# カスタムトークン有りモデル
PRETRAINED_MODEL_TOKEN_NAME = "daiki7069/t5_ad_emb"

# 転移学習後モデル
MODEL_NAME = "daiki7069/t5_ad_kw_epoch20_wo-body"
# MODEL_NAME = "daiki7069/t5_ad_kw_epoch8_wo-body"
# MODEL_NAME = "daiki7069/t5_ad_emb_epoch20_wo-body"
# MODEL_NAME = "daiki7069/t5_ad_emb2_epoch1_wo-body"
MODEL_REPO_TOKEN = "daiki7069/t5_ad_token"
MODEL_WITH_AA_TOKEN = "daiki7069/t5_ad_emb2"

# データセット一時保存場所
DATA_DIR = "/home/daiki_shibata/pj/ad-generation/RSL-PR-remote/ad_generation/__temp__"

# データセットのURL
# TEST_URL = "hf://datasets/daiki7069/camera-minimal/test.csv"
# DEV_URL = "hf://datasets/daiki7069/camera-minimal/dev.csv"
# TRAIN_URL = "hf://datasets/daiki7069/camera-minimal/train.csv"
TEST_URL = "hf://datasets/daiki7069/auto_classification/test.csv"
DEV_URL = "hf://datasets/daiki7069/auto_classification/dev.csv"
TRAIN_URL = "hf://datasets/daiki7069/auto_classification/train.csv"

# モデル一時保存場所
TEMP_MODEL_REPO = "daiki7069/t5_ad_kw"
