import os
import torch
import numpy as np
from torch.utils.data import Dataset

class TsvDatasetKwEmb(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        self.file_path = os.path.join(data_dir, type_path)

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, title, kw):
        # ニュースタイトル生成タスク用の入出力形式に変換する。
        input = f"{kw}"
        target = f"{title}"
        return input, target
    
    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                line = line.strip().split("\t")
                assert len(line) == 2
                assert len(line[0]) > 0
                assert len(line[1]) > 0

                kw = line[1]
                title = line[0]

                input, target = self._make_record(title, kw)    # TODO

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True,
                    padding="max_length", return_tensors="pt"
                )
                temp = tokenized_inputs["input_ids"][0,1].item()
                tokenized_inputs["input_ids"][0,1] = tokenized_inputs["input_ids"][0,0]
                tokenized_inputs["input_ids"][0,0] = temp

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True,
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
