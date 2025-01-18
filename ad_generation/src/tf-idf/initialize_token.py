import pandas as pd
import os
import MeCab
import ipadic
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_tfidf_reports(train_path: str):
    # データセット読み込み
    df_train_raw = pd.read_csv(train_path)

    # appealing_axis列の欠損値を8.0で埋める
    df_train_raw['appealing_axis'].fillna(8.0, inplace=True)

    # 0列目とasset_id列を削除
    df_train_raw = df_train_raw.drop(df_train_raw.columns[[0]], axis=1)  # 0列目を削除
    df_train_raw = df_train_raw.drop(columns=['asset_id'], errors='ignore')  # asset_id列を削除

    # appealing_axisをラベルとしてデータセットをクラス分け
    class_labels = df_train_raw['appealing_axis'].unique()
    class_datasets = {label: df_train_raw[df_train_raw['appealing_axis'] == label] for label in class_labels}

    # 各クラスについて単語ごとにTF-IDFを算出し、値の大きいものから順に単語をリスト
    tfidf_results = {}
    mecab = MeCab.Tagger(f"-O wakati {ipadic.MECAB_ARGS}")
    # mecab = MeCab.Tagger("-O wakati")  # MeCabの初期化 FIXME 

    for label, dataset in class_datasets.items():
        dataset['title_org'].fillna('', inplace=True)  # NaNを空文字列に置き換え
        # 形態素解析を行い、分かち書きにする
        dataset['title_org'] = dataset['title_org'].apply(lambda x: mecab.parse(x).strip())
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(dataset['title_org'])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        sorted_indices = tfidf_scores.argsort()[::-1]
        top_words = [(feature_names[i], tfidf_scores[i]) for i in sorted_indices]
        tfidf_results[label] = top_words

    # 出力
    for label, top_words in tfidf_results.items():
        print(f"{label}の上位単語:")
        for word, score in top_words[:10]:  # 上位10単語を表示
            print(f"単語: {word}, スコア: {score}")

    os.makedirs("../../top-tfidf-words", exist_ok=True)
    for label, top_words in tfidf_results.items():
        top_words_df = pd.DataFrame(top_words, columns=['単語', 'スコア'])
        top_words_df.to_csv(f'../../top-tfidf-words/top_words_{label}.csv', index=False, encoding='utf-8-sig')



if __name__ == "__main__":
    train_path = "hf://datasets/daiki7069/evaluetion/train_dec.csv"
    generate_tfidf_reports(train_path)
