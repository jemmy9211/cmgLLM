from bm25_pt import BM25
from datasets import load_dataset
import numpy as np
import json
with open("difftextV12.json") as f2:
    diffdata = json.load(f2)

ds = load_dataset("Maxscha/commitbench")

diffs = [sample['diff'] for sample in ds['test']]
commit_messages = [sample['message'] for sample in ds['test']]
bm25 = BM25(device='cuda')
bm25.index(diffs)

query = [diffdata[2]]

doc_scores = bm25.score_batch(query)
doc_scores = doc_scores.cpu().numpy()[0]

print(doc_scores)
print(type(doc_scores))
best_index = np.argmax(doc_scores)
print(doc_scores[best_index])
# 輸出最相關的 diff 及對應的 commit message
print(f"Most relevant diff: {diffs[best_index]}")
print(f"Corresponding commit message: {commit_messages[best_index]}")
print(f"Score: {doc_scores[best_index]}")
print('='*50)