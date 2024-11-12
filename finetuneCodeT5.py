import json
import nltk
from datasets import load_dataset, Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import warnings
from sklearn.model_selection import train_test_split
import random
import numpy as np

# 設定隨機種子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)
nltk.download('punkt', quiet=True)

# 載入資料
with open("msgtextV12.json") as f1:
    msgdata = json.load(f1)
with open("difftextV12.json") as f2:
    diffdata = json.load(f2)

assert len(msgdata) == len(diffdata), "Mismatch between message and diff data lengths."

torch.cuda.empty_cache()  # 清空初始 CUDA cache

# 構建訓練和測試數據集
data = [{"diff": diffdata[i], "message": msgdata[i]} for i in range(len(msgdata))]
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# 使用 Hugging Face 的 Dataset API
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# 載入 T5 Tokenizer 和模型
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m")
model.gradient_checkpointing_enable()

# 設置 use_cache=False 以兼容 gradient checkpointing
model.config.use_cache = False

# 定義數據處理函數
def preprocess_function(examples):
    # 構建輸入序列
    inputs = ["diff: " + diff for diff in examples["diff"]]
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        padding="max_length",  # 填充到最大長度
        truncation=True        # 截斷過長的序列
    )

    # 構建目標序列
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["message"],
            max_length=128,
            padding="max_length",  # 填充到最大長度
            truncation=True        # 截斷過長的序列
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 準備訓練數據
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 設置較小的批次大小來減少顯存佔用
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,  # 減小批次大小以避免CUDA out of memory
    per_device_eval_batch_size=2,   # 減小批次大小
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
    lr_scheduler_type="linear"
)

# 使用 Trainer API（不包含 BLEU 計算）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# 開始訓練
trainer.train()
torch.cuda.empty_cache()  # 清空訓練後的 CUDA cache

# 評估模型
results = trainer.evaluate()
torch.cuda.empty_cache()  # 清空評估後的 CUDA cache
print(f"Evaluation results: {results}")
