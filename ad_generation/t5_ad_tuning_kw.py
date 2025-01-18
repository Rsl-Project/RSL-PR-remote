import argparse
from src.tuning.dataset_handler_kw import DatasetHandlerKw
from src.util.env import TEST_URL, DEV_URL, TRAIN_URL
from src.util.env import PRETRAINED_MODEL_NAME
from src.util.params import args_dict
from src.util.env import TEMP_MODEL_REPO
from src.tuning.t5_fine_tuner_kw import T5FineTunerKw
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


def main():
    dataset = DatasetHandlerKw(test_url=TEST_URL, dev_url=DEV_URL, train_url=TRAIN_URL)

    dataset.csv2tsv()
    
    # トークナイザー（SentencePiece）モデルの読み込み
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)
    
    # 学習に用いるハイパーパラメータを設定する
    args_dict.update({
        "max_input_length":  512,  # 入力文の最大トークン数
        "max_target_length": 64,  # 出力文の最大トークン数
        "train_batch_size":  16,  # 訓練時のバッチサイズ
        "eval_batch_size":   16,  # テスト時のバッチサイズ
        "num_train_epochs":  20,  # 訓練するエポック数
        })
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        amp_level="O1",
        amp_backend="apex",
        gradient_clip_val=args.max_grad_norm,
    )

    # 転移学習の実行（GPUを利用すれば1エポック10分程度）
    model = T5FineTunerKw(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # 最終エポックのモデルを保存
    tokenizer.push_to_hub(f"{TEMP_MODEL_REPO}_epoch{args.num_train_epochs}_wo-body")
    model.push(f"{TEMP_MODEL_REPO}_epoch{args.num_train_epochs}_wo-body")

    del model

    dataset.remove_tsv()


if __name__ == "__main__":
    main()
