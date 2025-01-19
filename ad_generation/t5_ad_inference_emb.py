import csv
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, PhrasalConstraint
import argparse
from huggingface_hub import login

from src.util.env import MODEL_NAME
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
    MAX_TARGET_LENGTH = 16

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
    with open('./app3_kw2title_t5-ep20-wobody.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "kw", "appealing_axis", "variation", "generated_title"])
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
            for app_axis in range(8):
                if app_axis == 0:
                    constraint_words = ["格安", "予約", "安値"]
                elif app_axis == 1:
                    constraint_words = ["無料", "試し", "体験"]
                elif app_axis == 2:
                    constraint_words = ["すぐ", "なら", "今日"]
                elif app_axis == 3:
                    constraint_words = ["徒歩", "近く", "周辺"]
                elif app_axis == 4:
                    constraint_words = ["予約", "個室", "ok"]
                elif app_axis == 5:
                    constraint_words = ["限定", "向け", "キャンペーン"]
                elif app_axis == 6:
                    constraint_words = ["比較", "一覧", "公式"]
                elif app_axis == 7:
                    constraint_words = ["ランキング", "最新", "人気"]
                else:
                    raise ValueError('appaealing_axisは0~7の値です.')
                
                constraints = [
                    PhrasalConstraint(
                        tokenizer(constraint_words[0], add_special_tokens=True).input_ids,
                    ),
                    PhrasalConstraint(
                        tokenizer(constraint_words[1], add_special_tokens=True).input_ids,
                    ),
                    PhrasalConstraint(
                        tokenizer(constraint_words[2], add_special_tokens=True).input_ids,
                    ),
                ]
                outputs_kw = trained_model.generate(
                    input_ids=input_ids_kw, attention_mask=input_mask_kw,
                    constraints=constraints,
                    max_length=MAX_TARGET_LENGTH,
                    temperature=3.0,          # 生成にランダム性を入れる温度パラメータ default=2.0
                    num_beams=10,             # ビームサーチの探索幅 default=10
                    num_return_sequences=1,  # 生成する文の数 default=10
                    repetition_penalty=5.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ default=1.5
                )

                # 生成されたトークン列を文字列に変換する
                generated_titles_kw = [tokenizer.decode(ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                                    for ids in outputs_kw]
                
                # 生成されたタイトルを表示する
                for i, title in enumerate(generated_titles_kw):
                    print(f"[kw{index}] asset_id:{id} appealing_axis:{app_axis} {i+1:2}. {title}")
                    writer.writerow([id, inputs_kw[0], app_axis, i, generated_titles_kw[0]])
            
            index += 1


    ### lp_meta_descriptionから生成
    with open('./app3_description2title_t5-ep20-wobody.csv', 'w') as f:
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
            for app_axis in range(8):
                if app_axis == 0:
                    constraint_words = ["格安", "予約", "安値"]
                elif app_axis == 1:
                    constraint_words = ["無料", "試し", "体験"]
                elif app_axis == 2:
                    constraint_words = ["すぐ", "なら", "今日"]
                elif app_axis == 3:
                    constraint_words = ["徒歩", "近く", "周辺"]
                elif app_axis == 4:
                    constraint_words = ["予約", "個室", "ok"]
                elif app_axis == 5:
                    constraint_words = ["限定", "向け", "キャンペーン"]
                elif app_axis == 6:
                    constraint_words = ["比較", "一覧", "公式"]
                elif app_axis == 7:
                    constraint_words = ["ランキング", "最新", "人気"]
                else:
                    raise ValueError('appaealing_axisは0~7の値です.')
                
                constraints = [
                    PhrasalConstraint(
                        tokenizer(constraint_words[0], add_special_tokens=False).input_ids,
                    ),
                    PhrasalConstraint(
                        tokenizer(constraint_words[1], add_special_tokens=False).input_ids,
                    ),
                    PhrasalConstraint(
                        tokenizer(constraint_words[2], add_special_tokens=False).input_ids,
                    ),
                ]

                outputs_str = trained_model.generate(
                    input_ids=input_ids_str, attention_mask=input_mask_str,
                    constraints=constraints,
                    max_length=MAX_TARGET_LENGTH,
                    temperature=3.0,          # 生成にランダム性を入れる温度パラメータ default=2.0
                    num_beams=10,             # ビームサーチの探索幅 default=10
                    num_return_sequences=1,  # 生成する文の数 default=10
                    repetition_penalty=5.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ default=1.5
                )

                # 生成されたトークン列を文字列に変換する
                generated_titles_str = [tokenizer.decode(ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                                    for ids in outputs_str]
                for i, title in enumerate(generated_titles_str):
                    print(f"[str{index}], asset_id:{id} appealing_axis:{app_axis} {i+1:2}. {title}")
                    writer.writerow([id, inputs_str[0], app_axis, i, generated_titles_str[0]])

            index += 1
    


if __name__ == "__main__":
    main()
