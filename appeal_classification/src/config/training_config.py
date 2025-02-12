class TrainingConfig:
    output_dir: str = "roberta-classification-model"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16    # default: 16
    per_device_eval_batch_size: int = 16    # default: 16
    num_train_epochs: int = 20              # default: 20
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    push_to_hub: bool = True
    hub_model_name: str = "daiki7069/temp_model_roberta_log"
    