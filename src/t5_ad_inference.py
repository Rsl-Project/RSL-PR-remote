import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import argparse
from huggingface_hub import login

from env import MODEL_NAME, TEMP_MODEL_REPO
from params import args_dict
from normalization import normalize_text

def main():
    # huggingfaceログイン
    # login()

    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(TEMP_MODEL_REPO, is_fast=True)
    # tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained(TEMP_MODEL_REPO)
    # trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    # trained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    # ハイパーパラメーターの設定
    args = argparse.Namespace(**args_dict)

    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

    def preprocess_body(text):
        return normalize_text(text.replace("\n", " "))

    # 推論モード設定
    trained_model.eval()

    # 入力文字列
    with open("./input.txt", "r") as f:
        body = f.read()
        f.close()

    # 前処理とトークナイズを行う
    inputs = [preprocess_body(body)]
    batch = tokenizer.batch_encode_plus(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True,
        padding="longest", return_tensors="pt")

    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    # 生成処理を行う
    outputs = trained_model.generate(
        input_ids=input_ids, attention_mask=input_mask,
        max_length=MAX_TARGET_LENGTH,
        temperature=3.0,          # 生成にランダム性を入れる温度パラメータ default=2.0
        num_beams=10,             # ビームサーチの探索幅 default=10
        diversity_penalty=3.0,    # 生成結果の多様性を生み出すためのペナルティ default=3.0
        num_beam_groups=10,       # ビームサーチのグループ数 default=10
        num_return_sequences=10,  # 生成する文の数 default=10
        repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ default=1.5
    )

    # 生成されたトークン列を文字列に変換する
    generated_titles = [tokenizer.decode(ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                        for ids in outputs]

    # 入力文字列を表示
    print(f"input: {body}")

    # 生成されたタイトルを表示する
    for i, title in enumerate(generated_titles):
        print(f"{i+1:2}. {title}")


if __name__ == "__main__":
    main()
