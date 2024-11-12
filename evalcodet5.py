from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datasets import load_dataset
from tqdm import tqdm

# 下載 NLTK 所需的資源
nltk.download('punkt', quiet=True)

# 載入 Tokenizer 和微調後的模型
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
model = T5ForConditionalGeneration.from_pretrained("./results/checkpoint-81592")

# 將模型移動到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 設置模型為評估模式
model.eval()

# 載入 CommitBench 數據集
ds = load_dataset("Maxscha/commitbench")
CommitBenchdiffs = ds['test']['diff'][:20000]
CommitBenchcommit_messages = ds['test']['message'][:20000]

all_references = []
all_candidates = []

# 遍歷取出的 20,000 筆數據，生成預測並計算 BLEU 分數
for diff, message in tqdm(zip(CommitBenchdiffs, CommitBenchcommit_messages), total=len(CommitBenchdiffs), desc="Processing"):
    # 編碼輸入，並將其移動到 GPU
    inputs = tokenizer("diff: " + diff, return_tensors="pt", max_length=512, truncation=True).to(device)
    # 使用模型生成預測
    with torch.no_grad():  # 禁用梯度計算以加速推理
        outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    # 解碼生成的輸出
    candidate = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # BLEU 分數需要標記化的文本
    reference = [nltk.word_tokenize(message.lower())]  # 期望的 commit message
    candidate_tokens = nltk.word_tokenize(candidate.lower())  # 生成的 commit message

    all_references.append(reference)
    all_candidates.append(candidate_tokens)

# 計算 BLEU 分數
smooth_fn = SmoothingFunction().method2  # 使用平滑函數
bleu_score = corpus_bleu(all_references, all_candidates, smoothing_function=smooth_fn)

print(f"Corpus BLEU-4 score: {bleu_score}")
