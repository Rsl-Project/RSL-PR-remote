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
    MODEL_NAME = "daiki7069/t5_ad_kw_epoch20_wo-body"
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

    # lp_meta_descriptionから単語を抜き出すためのtf-idf単語リストを読み込む
    tfidf_words_df_0 = pd.read_csv('./top-tfidf-words/top_words_0.csv')
    tfidf_words_df_1 = pd.read_csv('./top-tfidf-words/top_words_1.csv')
    tfidf_words_df_2 = pd.read_csv('./top-tfidf-words/top_words_2.csv')
    tfidf_words_df_3 = pd.read_csv('./top-tfidf-words/top_words_3.csv')
    tfidf_words_df_4 = pd.read_csv('./top-tfidf-words/top_words_4.csv')
    tfidf_words_df_5 = pd.read_csv('./top-tfidf-words/top_words_5.csv')
    tfidf_words_df_6 = pd.read_csv('./top-tfidf-words/top_words_6.csv')
    tfidf_words_df_7 = pd.read_csv('./top-tfidf-words/top_words_7.csv')
    tfidf_words_0 = tfidf_words_df_0['word'].tolist()
    tfidf_words_1 = tfidf_words_df_1['word'].tolist()
    tfidf_words_2 = tfidf_words_df_2['word'].tolist()
    tfidf_words_3 = tfidf_words_df_3['word'].tolist()
    tfidf_words_4 = tfidf_words_df_4['word'].tolist()
    tfidf_words_5 = tfidf_words_df_5['word'].tolist()
    tfidf_words_6 = tfidf_words_df_6['word'].tolist()
    tfidf_words_7 = tfidf_words_df_7['word'].tolist()

    def preprocess_body(text):
        s = normalize_text(text.replace("\n", " "))
        return s.replace("/", " ")
    
    ### kwから生成
    with open('./firs_app_kw2title_t5-ep20-wobody.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "kw", "appealing_axis", "variation", "generated_title", "constraint_word"])

        index = 0
        for id, record, description in zip(df["asset_id"].tolist(), df["kw"].tolist(), df["lp_meta_description"].tolist()):
            # 前処理とトークナイズを行う
            inputs_kw_pre = [preprocess_body(record)]

            ### 生成処理を行う
            for app_axis in range(8):
                if app_axis == 0:
                    constraint_word = "格安"
                elif app_axis == 1:
                    constraint_word = "無料"
                elif app_axis == 2:
                    constraint_word = "すぐ"     
                elif app_axis == 3:
                    constraint_word = "徒歩"                 
                elif app_axis == 4:
                    constraint_word = "保険"               
                elif app_axis == 5:
                    constraint_word = "限定"                  
                elif app_axis == 6:
                    constraint_word = "公式"
                elif app_axis == 7:
                    constraint_word = "2022"                   
                else:
                    raise ValueError('appaealing_axisは0~7の値です.')
                
                inputs_kw = [constraint_word + " " + inputs_kw_pre[0]]
                print(inputs_kw[0])
                batch_kw = tokenizer.batch_encode_plus(
                    inputs_kw, max_length=MAX_SOURCE_LENGTH, truncation=True,
                    padding="longest", return_tensors="pt")
                input_ids_kw = batch_kw['input_ids']
                input_mask_kw = batch_kw['attention_mask']
                if USE_GPU:
                    input_ids_kw = input_ids_kw.cuda()
                    input_mask_kw = input_mask_kw.cuda()

                # constraints = [
                #     PhrasalConstraint(
                #         tokenizer(constraint_word, add_special_tokens=False).input_ids,
                #     ),
                # ]
                print(input_ids_kw)
                outputs_kw = trained_model.generate(
                    input_ids=input_ids_kw, attention_mask=input_mask_kw,
                    # decoder_input_ids=torch.tensor(tokenizer(constraint_word, add_special_tokens=False).input_ids),
                    # constraints=constraints,
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
                    writer.writerow([id, inputs_kw[0], app_axis, i, generated_titles_kw[0], constraint_word])
            
            index += 1


    ### lp_meta_descriptionから生成
    with open('./first_app_description2title_t5-ep20-wobody.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "lp_meta_description", "variation", "generated_title", "constraint_word"])
        index = 0
        for id, description in zip(df["asset_id"].tolist(), df["lp_meta_description"].tolist()):
            # 前処理とトークナイズを行う
            inputs_str_pre = [preprocess_body(description)]

            ### 生成処理を行う
            for app_axis in range(8):
                if app_axis == 0:
                    constraint_word = "格安"
                elif app_axis == 1:
                    constraint_word = "無料"
                elif app_axis == 2:
                    constraint_word = "すぐ"     
                elif app_axis == 3:
                    constraint_word = "徒歩"                 
                elif app_axis == 4:
                    constraint_word = "保険"               
                elif app_axis == 5:
                    constraint_word = "限定"                  
                elif app_axis == 6:
                    constraint_word = "公式"
                elif app_axis == 7:
                    constraint_word = "2022" 
                else:
                    raise ValueError('appaealing_axisは0~7の値です.')
                
                inputs_str = [constraint_word + " " + inputs_str_pre[0]]
                print(inputs_str[0])
                batch_str = tokenizer.batch_encode_plus(
                    inputs_str, max_length=MAX_SOURCE_LENGTH, truncation=True,
                    padding="longest", return_tensors="pt")
                input_ids_str = batch_str['input_ids']
                input_mask_str = batch_str['attention_mask']
                if USE_GPU:
                    input_ids_str = input_ids_str.cuda()
                    input_mask_str = input_mask_str.cuda()

                # constraints = [
                #     PhrasalConstraint(
                #         tokenizer(constraint_word, add_special_tokens=False).input_ids,
                #     ),
                # ]

                outputs_str = trained_model.generate(
                    input_ids=input_ids_str, attention_mask=input_mask_str,
                    # decoder_input_ids=torch.tensor(tokenizer(constraint_word, add_special_tokens=False).input_ids),
                    # constraints=constraints,
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
                    writer.writerow([id, inputs_str[0], app_axis, i, generated_titles_str[0], constraint_word])

            index += 1
    


if __name__ == "__main__":
    main()
