import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn

df_true = pd.read_csv("hf://datasets/daiki7069/auto_classification/train.csv")
appealing_axis_true = df_true.appealing_axis.tolist()

df_pred = pd.read_csv("hf://datasets/daiki7069/auto_classification/train_pred.csv")
appealing_axis_pred = df_pred.label.tolist()

name = [0, 1, 2, 3, 4, 5, 6, 7, 8]
val_mat = confusion_matrix(appealing_axis_true, appealing_axis_pred, labels=name)
acc_score = accuracy_score(appealing_axis_true, appealing_axis_pred)
rec_score = recall_score(appealing_axis_true, appealing_axis_pred, average=None)
pre_score = precision_score(appealing_axis_true, appealing_axis_pred, average=None)
micro_f1_score = f1_score(appealing_axis_true, appealing_axis_pred, average=None)
macro_f1_score = f1_score(appealing_axis_true, appealing_axis_pred, average="macro")
print(val_mat)
print(f"Accuracy: {acc_score}")
print(f"Recall [class0, ..., class8]: {rec_score}")
print(f"Precision [class0, ..., class8]: {pre_score}")
print(f"micro-F1 score [class0, ..., class8]: {micro_f1_score}")
print(f"macro-F1 score: {macro_f1_score}")


plt.figure(figsize=(16, 16))
seaborn.heatmap(val_mat, square=True, cbar=True, annot=True, cmap='Blues')
plt.title("classifier")
plt.xlabel("predicted")
plt.ylabel("generated")
plt.savefig("classifier.png")
