import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import \
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, pipeline, DataCollatorWithPadding
from src.config.dataset_config import DatasetConfig
from src.config.pre_trained_model_config import PreTrainedModelConfig
from src.config.training_config import TrainingConfig
from src.preprocess import Preprocessor


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # データセットの前処理
    preprocessor = Preprocessor()
    df_train, df_dev, df_test = preprocessor.preprocess(mode=0)

    # 訓練データと評価データを結合
    df_train = pd.concat([df_train, df_dev], ignore_index=True)
    print(f"結合後の訓練データ数: {len(df_train)}")

    # 学習, 評価データからラベルを取り出す
    unique_labels = df_train.label.unique()

    # id からラベルへの相互変換辞書を用意する
    id2label = dict([(id, label) for id, label in enumerate(unique_labels)])
    label2id = dict([(label, id) for id, label in enumerate(unique_labels)])

    # 学習データのラベルを id で置き換える
    df_train.label = df_train.label.map(label2id.get)

    # 学習データを 8:2 の割合で学習に使用するデータと評価用データに分ける
    df_train, df_dev = train_test_split(df_train, test_size=0.2)
    print(f"# of train = {df_train.shape[0]}, # of eval = {df_dev.shape[0]}")
    
    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset = Dataset.from_pandas(df_dev)
    test_dataset = Dataset.from_pandas(df_test)

    pre_trained_model_config = PreTrainedModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_model_config.tokenizer_name, 
        use_fast=False,
        padding=True,
        truncation=True,
        max_length=512
    )
    tokenizer.do_lower_case = True  # トークナイザーの設定読み込みのバグによる対応

    def preprocess_function(examples):
        # データセットに含まれる text を取り出して tokenizeする
        tokenized = tokenizer(examples["title_org"])
        tokenized['label'] = np.array(examples["label"]).astype(int)  # リストをNumPy配列に変換
        # print("=== トークナイズ結果 ===")
        # print(f"入力テキスト: {examples['title_org'][:3]}")  # 最初の3件のみ表示
        # print(f"トークンID: {tokenized['input_ids'][:3]}")
        # print(f"アテンションマスク: {tokenized['attention_mask'][:3]}")
        return tokenized

    # 学習用、評価用、検証用の 3つのデータセットを用意
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    print("データ", tokenized_train_dataset)    # TODO: remove

    model = AutoModelForSequenceClassification.from_pretrained(
        pre_trained_model_config.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    training_config = TrainingConfig()
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        num_train_epochs=training_config.num_train_epochs,
        weight_decay=training_config.weight_decay,
        evaluation_strategy=training_config.evaluation_strategy,
        save_strategy=training_config.save_strategy,
        load_best_model_at_end=training_config.load_best_model_at_end,
        push_to_hub=training_config.push_to_hub,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)    # TODO
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # TrainerにDataCollatorを渡す
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("学習が終了しました。")

    model.to('cpu') # TODO

    # パイプラインの設定
    classifier = pipeline("text-classification", tokenizer=tokenizer, model=model)

    # ここで実際の推論を実行
    result = df_test.title_org.apply(classifier)

    # 結果を表示するために色々変換
    labels = [r[0]['label'] for r in result]
    scores = [r[0]['score'] for r in result]
    df_result = pd.DataFrame({
        'label': labels,
        'score': scores,
    })
    print("推論結果:")
    print(df_result)


if __name__ == "__main__":
    main()
