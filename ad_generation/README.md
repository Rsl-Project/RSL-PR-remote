## 広告文生成
広告文生成用の処理です。
#### 使用モデル
- t5-base-japanese
    - 広告文データセット`CAMERA`を転移学習して用いる
#### 使用データセット
- CAMERA-minimal
    - LP画像なしの広告データセット


## ディレクトリ構成
- t5_ad_inference.py
    - 広告文生成(推論)を行う
    - 使用するモデルは`ad_generation/src/util/env.py`内の`MODEL_NAME`に記載する
- t5_ad_generation.py
    - 転移学習を行う
    - 学習用データセットは`ad_generation/src/util/env.py`内の`データセットのURL`に記載する
- src
    - tuning
        - 転移学習時に用いるモジュール
    - inference
        - 推論時に用いるモジュール
    - util
        - 共通のモジュール
    - old
        - 以前のモジュールやテストコード


## 環境構築
```bash
pip install -r requirement.txt
```

## 実行方法
```bash
cd ad_generation
```
#### 学習時
```bash
python t5_ad_tuning.py
```
#### 推論時
```bash
python t5_ad_inference.py
```
