import pandas as pd
from transformers import \
    AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.preprocess import Preprocessor


def inference(url):
    
    df = pd.read_csv(url)
    # df['title_org'] = df['title_org'].fillna(" ")

    MODEL_NAME = "daiki7069/temp_model_class_epoch20_batch16"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.do_lower_case = True  # トークナイザーの設定読み込みのバグによる対応
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=9)

    # model.to('cpu') # TODO

    # パイプラインの設定
    classifier = pipeline("text-classification", tokenizer=tokenizer, model=model)

    # ここで実際の推論を実行
    # result = df.title_org.apply(classifier)
    result = df.generated_title.apply(classifier)
    print(result)

    # 結果を表示するために色々変換
    labels = [r[0]['label'] for r in result]
    scores = [r[0]['score'] for r in result]
    titles = df.values.tolist()
    df_result = pd.DataFrame({
        'label': labels,
        'score': scores,
        'title': titles,
    })
    print("推論結果:")
    print(df_result)
    df_result.to_csv("./firs_app_kw2title_t5-ep20-wobody_pred.csv")
    return 


if __name__ == "__main__":
    inference("hf://datasets/daiki7069/generated/firs_app_kw2title_t5-ep20-wobody.csv")