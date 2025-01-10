# RSL-PR-remote
- appeal_classification
    - RoBERTaを用いた訴求軸判定器
- ad_generation
    - T5-base-japaneseを用いた広告文生成

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