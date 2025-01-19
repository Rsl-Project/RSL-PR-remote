import os
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from src.config.training_config import TrainingConfig

class LearningRateLogger(TrainerCallback):
    def __init__(self):
        super().__init__()

        self.train_epoch = []
        self.train_loss = []
        self.eval_epoch = []
        self.eval_loss = []

        self.eval_accuracy = []
        self.eval_f1 = []
        self.eval_precision = []
        self.eval_recall = []


    # def on_step_end(self, args, state, control, **kwargs):
    #     print(state.log_history)


    def on_log(self, args, state, control, **kwargs):
        t_ep = state.log_history[-1].get('epoch')
        t_ls = state.log_history[-1].get('loss')

        # 欠損値を除外
        if not(t_ep == None or t_ls == None):
            self.train_epoch.append(t_ep)
            self.train_loss.append(t_ls)
        print("=====train========")
        print(state.log_history[-1])
        print("=================")

    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        e_ep = state.log_history[-1].get('epoch')
        e_ls = state.log_history[-1].get('eval_loss')
        # 欠損値を除外
        if not(e_ep == None or e_ls == None):
            self.eval_epoch.append(e_ep)
            self.eval_loss.append(e_ls)
        
        self.eval_accuracy.append(state.log_history[-1].get('eval_accuracy'))
        self.eval_f1.append(state.log_history[-1].get('eval_f1'))
        self.eval_precision.append(state.log_history[-1].get('eval_precision'))
        self.eval_recall.append(state.log_history[-1].get('eval_recall'))
        print("=====eval========")
        print(state.log_history[-1])
        print("=================")


    # def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     MODEL_REPO = "daiki7069/temp_model_class"
    #     model_path = f"{MODEL_REPO}/epoch-{int(state.epoch)}/checkpoint-{state.global_step}"
    #     # self.tokenizer.push_to_hub(model_path)
    #     # self.model.push_to_hub(model_path)



    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        end_epoch = int(self.eval_epoch[-1])
        # ディレクトリが存在しない場合は作成
        os.makedirs(f'graph/epoch{end_epoch}', exist_ok=True)

        # 学習率曲線を描画
        plt.plot(self.train_epoch, self.train_loss, label="train_data")
        plt.plot(self.eval_epoch, self.eval_loss, label="eval_data")
        plt.plot()
        plt.title(f"Learning Rate Curve epoch_{end_epoch} batch_{TrainingConfig.per_device_train_batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend(loc=0)
        plt.savefig(f'graph/epoch{end_epoch}/curb_epoch{end_epoch}_batch{TrainingConfig.per_device_train_batch_size}.png')
        plt.clf()

        # F値などを表示
        plt.plot(self.eval_epoch, self.eval_accuracy, label="eval_accuracy")
        plt.plot(self.eval_epoch, self.eval_f1, label="eval_f1")
        plt.plot(self.eval_epoch, self.eval_precision, label="eval_precision")
        plt.plot(self.eval_epoch, self.eval_recall, label="eval_recall")
        plt.title(f"Info epoch_{end_epoch} batch_{TrainingConfig.per_device_train_batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.grid()
        plt.legend(loc=0)
        plt.savefig(f'graph/epoch{end_epoch}/info_epoch{end_epoch}_batch{TrainingConfig.per_device_train_batch_size}.png')
        plt.clf()
