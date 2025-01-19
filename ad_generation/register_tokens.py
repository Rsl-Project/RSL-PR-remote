import torch
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from src.util.env import PRETRAINED_MODEL_NAME \
    ,PRETRAINED_MODEL_TOKEN_NAME



def main():
    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)
    # 学習済みモデル
    model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_NAME).encoder

    ### 訴求トークンの初期化
    # 単語追加
    # 現状special_tokens=Trueを設定しないとエラーコードが返ってくるようです。
    tokenizer.add_tokens([f"[aa{i}]" for i in range(9)], special_tokens=True)
    # tokenizer.add_tokens(["[aa0]", "[aa1]", "[aa2]", "[aa3]", "[aa4]", "[aa5]", "[aa6]", "[aa7]", "[aa8]"], special_tokens=True)

    # モデルに訴求トークンの埋め込みベクトルを追加
    pre_size = model.get_input_embeddings().weight.data.size()[0]
    model.resize_token_embeddings(pre_size+9)

    # ベクトルの初期化
    def _init_token(words):
        for i, word in zip(range(len(words)), words):
            word_ids = tokenizer.encode(word)
            word_ids.pop(-1)
            # print(word_ids)
            # print(tokenizer.decode(word_ids))
            word_embeds = model(torch.tensor(word_ids).unsqueeze(0))
            # print(word_embeds.last_hidden_state)
            word_embed = torch.mean(word_embeds.last_hidden_state.squeeze(), dim=0)
            # print(word_embed)
            if i == 0:
                embeds = word_embed
            else:
                embeds = embeds + word_embed
        return  embeds / len(word)  # 平均化
        
    # aa0_words = ["格安", "予約", "安値"]
    # aa1_words = ["無料", "試し", "体験"]
    # aa2_words = ["すぐ", "なら", "今日"]
    # aa3_words = ["徒歩", "近く", "周辺"]
    # aa4_words = ["予約", "個室", "ok"]
    # aa5_words = ["限定", "向け", "キャンペーン"]
    # aa6_words = ["比較", "一覧", "公式"]
    # aa7_words = ["ランキング", "最新", "人気"]
    aa_words = [["格安", "予約", "安値"],
                ["無料", "試し", "体験"],
                ["すぐ", "なら", "今日"],
                ["徒歩", "近く", "周辺"],
                ["予約", "個室", "ok"],
                ["限定", "向け", "キャンペーン"],
                ["比較", "一覧", "公式"],
                ["ランキング", "最新", "人気"]]
    for i, aa_word in zip(range(8), aa_words):
        model.get_input_embeddings().weight.data[32100+i,:] = _init_token(aa_word)
    # model.get_input_embeddings().weight.data[32108,:] = # [aa8]
    # model.get_input_embeddings().weight.data[32107,:] = # [aa7]
    # model.get_input_embeddings().weight.data[32106,:] = # [aa6]
    # model.get_input_embeddings().weight.data[32105,:] = # [aa5]
    # model.get_input_embeddings().weight.data[32104,:] = # [aa4]
    # model.get_input_embeddings().weight.data[32103,:] = # [aa3]
    # model.get_input_embeddings().weight.data[32102,:] = # [aa2]
    # model.get_input_embeddings().weight.data[32101,:] = # [aa1]
    # model.get_input_embeddings().weight.data[32100,:] = _init_token(aa0_words)  # [aa0]
    

    tokenizer.push_to_hub(PRETRAINED_MODEL_TOKEN_NAME)
    model.push_to_hub(PRETRAINED_MODEL_TOKEN_NAME)



if __name__ == "__main__":
    main()