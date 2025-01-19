import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from src.util.env import DATA_DIR, PRETRAINED_MODEL_TOKEN_NAME

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# GPU利用有無
USE_GPU = torch.cuda.is_available()

# 各種ハイパーパラメータ
args_dict = dict(
    data_dir=DATA_DIR,  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_TOKEN_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_TOKEN_NAME,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    max_input_length=512,
    max_target_length=64,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=4,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    max_grad_norm=1.0,
    seed=42,
)