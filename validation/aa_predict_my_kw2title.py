import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn
import csv


df_pred = pd.read_csv("./my_firs_app_kw2title_t5-ep20-wobody.csv")
appealing_axis_pred = df_pred.label.tolist()

# 0-7の数字をrepeat_count回繰り返したリストを作成
appealing_axis_true = [ i%8 for i in range(len(appealing_axis_pred))]
print(appealing_axis_true)

name = [0, 1, 2, 3, 4, 5, 6, 7]
val_mat = confusion_matrix(appealing_axis_true, appealing_axis_pred, labels=name)
acc_score = accuracy_score(appealing_axis_true, appealing_axis_pred)
rec_score = recall_score(appealing_axis_true, appealing_axis_pred, average=None)
pre_score = precision_score(appealing_axis_true, appealing_axis_pred, average=None)
micro_f1_score = f1_score(appealing_axis_true, appealing_axis_pred, average=None)
macro_f1_score = f1_score(appealing_axis_true, appealing_axis_pred, average="macro")
print(val_mat)
print(f"Accuracy: {acc_score}")
print(f"Recall [class0, ..., class7]: {rec_score}")
print(f"Precision [class0, ..., class7]: {pre_score}")
print(f"micro-F1 score [class0, ..., class7]: {micro_f1_score}")
print(f"macro-F1 score: {macro_f1_score}")


plt.figure(figsize=(16, 16))
seaborn.heatmap(val_mat, square=True, cbar=True, annot=True, cmap='Blues')
plt.title("classifier")
plt.xlabel("predicted")
plt.ylabel("generated")
plt.savefig("my_first_app_kw2title.png")

with open("./my_first_app_kw2title.txt", "w") as f:
    f.write(f"{val_mat}\n")
    f.write(f"Accuracy: {acc_score}\n")
    f.write(f"Recall [class0, ..., class7]: {rec_score}\n")
    f.write(f"Precision [class0, ..., class7]: {pre_score}\n")
    f.write(f"micro-F1 score [class0, ..., class7]: {micro_f1_score}\n")
    f.write(f"macro-F1 score: {macro_f1_score}\n")
    f.close()
