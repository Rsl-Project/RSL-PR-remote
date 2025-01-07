import os

from env import DATA_DIR

def remove_data(self) -> None:
    os.remove(f"{DATA_DIR}/dev.tsv")
    os.remove(f"{DATA_DIR}/test.tsv")
    os.remove(f"{DATA_DIR}/train.tsv")

    os.rmdir(DATA_DIR)