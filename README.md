# RSL-PR-remote
- appeal_classification
    - RoBERTaを用いた訴求軸判定器
- ad_generation
    - T5-base-japaneseを用いた広告文生成
- validation
    - 訴求軸判定器と広告文生成モデルの性能評価

## Directory Structure
```
.
├── README.md
├── ad_generation
│   ├── README.md
│   ├── input.txt
│   ├── requirements.txt
│   ├── src
│   │   ├── inference
│   │   ├── old
│   │   │   ├── load_data.py
│   │   │   ├── make_dataset.py
│   │   │   └── remove_data.py
│   │   ├── tuning
│   │   │   ├── __init__.py
│   │   │   ├── dataset_handler.py
│   │   │   ├── t5_fine_tuner.py
│   │   │   └── tsv_dataset.py
│   │   └── util
│   │       ├── __init__.py
│   │       ├── env.py
│   │       ├── normalization.py
│   │       └── params.py
│   ├── t5_ad_inference.py
│   └── t5_ad_tuning.py
└── appeal_classification
    ├── README.md
    ├── main.py
    └── src
        └── __init__.py
```

## ad_generation
### Prefix constraint decoding (PCD)
オリジナルの訴求トークンを用いて生成時の制御性を向上させたモデル.  
訴求トークンは[AA0]~[AA8].
- t5_ad_tuning_kw_emb.py
    - Fine-tningを行うスクリプト
    - input: 訴求トークン(1文字目) + 広告キーワード
    - target: 広告文
- t5_ad_inference_emb.py
    - 推論を行うスクリプト
    - input: 訴求トークン(1文字目) + 広告キーワード
        - 例: [AA0] 保険 65歳
    - output: 広告文

### 
