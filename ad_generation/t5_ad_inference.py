import csv
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import argparse
from huggingface_hub import login

from src.util.env import MODEL_NAME, TEMP_MODEL_REPO
from src.util.params import args_dict
from src.util.normalization import normalize_text

def main():
    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)
    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    # ハイパーパラメーターの設定
    args = argparse.Namespace(**args_dict)
    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    # MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数
    MAX_TARGET_LENGTH = 64

    # 推論モード設定
    trained_model.eval()

    # 入力
    df = pd.read_csv("hf://datasets/daiki7069/camera-minimal/test.csv")
    df = df.drop(columns=[
        "title_org",
        "parsed_full_text_annotation",
    ])
    df["kw"] = df["kw"].map(lambda s: s.replace(" ", "/"))

    def preprocess_body(text):
        s = normalize_text(text.replace("\n", " "))
        return s.replace("/", " ")
    
    ### kwから生成
    with open('./kw2title_t5-ep20-wobody.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "kw", "variation", "generated_title"])
        index = 0
        for id, record in zip(df["asset_id"].tolist(), df["kw"].tolist()):
            # 前処理とトークナイズを行う
            inputs_kw = [preprocess_body(record)]
            # print(inputs_kw)
            batch_kw = tokenizer.batch_encode_plus(
                inputs_kw, max_length=MAX_SOURCE_LENGTH, truncation=True,
                padding="longest", return_tensors="pt")
            input_ids_kw = batch_kw['input_ids']
            input_mask_kw = batch_kw['attention_mask']
            if USE_GPU:
                input_ids_kw = input_ids_kw.cuda()
                input_mask_kw = input_mask_kw.cuda()

            ### 生成処理を行う
            outputs_kw = trained_model.generate(
                input_ids=input_ids_kw, attention_mask=input_mask_kw,
                max_length=MAX_TARGET_LENGTH,
                temperature=3.0,          # 生成にランダム性を入れる温度パラメータ default=2.0
                num_beams=10,             # ビームサーチの探索幅 default=10
                diversity_penalty=3.0,    # 生成結果の多様性を生み出すためのペナルティ default=3.0
                num_beam_groups=10,       # ビームサーチのグループ数 default=10
                num_return_sequences=1,  # 生成する文の数 default=10
                repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ default=1.5
            )

            # 生成されたトークン列を文字列に変換する
            generated_titles_kw = [tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                                for ids in outputs_kw]
            
            # 生成されたタイトルを表示する
            for i, title in enumerate(generated_titles_kw):
                print(f"[kw{index}] asset_id:{id}, {i+1:2}. {title}")
                writer.writerow([id, inputs_kw[0], i, generated_titles_kw[0]])
            
            index += 1


    ### lp_meta_descriptionから生成
    with open('./description2title_t5-ep20-wobody.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "lp_meta_description", "variation", "generated_title"])
        index = 0
        for id, record in zip(df["asset_id"].tolist(), df["lp_meta_description"].tolist()):
            # 前処理とトークナイズを行う
            inputs_str = [preprocess_body(record)]
            # print(inputs_str)
            batch_str = tokenizer.batch_encode_plus(
                inputs_str, max_length=MAX_SOURCE_LENGTH, truncation=True,
                padding="longest", return_tensors="pt")
            input_ids_str = batch_str['input_ids']
            input_mask_str = batch_str['attention_mask']
            if USE_GPU:
                input_ids_str = input_ids_str.cuda()
                input_mask_str = input_mask_str.cuda()

            ### 生成処理を行う
            outputs_str = trained_model.generate(
                input_ids=input_ids_str, attention_mask=input_mask_str,
                max_length=MAX_TARGET_LENGTH,
                temperature=3.0,          # 生成にランダム性を入れる温度パラメータ default=2.0
                num_beams=10,             # ビームサーチの探索幅 default=10
                diversity_penalty=3.0,    # 生成結果の多様性を生み出すためのペナルティ default=3.0
                num_beam_groups=10,       # ビームサーチのグループ数 default=10
                num_return_sequences=1,  # 生成する文の数 default=10
                repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ default=1.5
            )

            # 生成されたトークン列を文字列に変換する
            generated_titles_str = [tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                                for ids in outputs_str]
            for i, title in enumerate(generated_titles_str):
                print(f"[str{index}], asset_id:{id} {i+1:2}. {title}")
                writer.writerow([id, inputs_kw[0], i, generated_titles_str[0]])

            index += 1
    


if __name__ == "__main__":
    main()
