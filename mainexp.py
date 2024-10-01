import json
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import threading
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
from bm25_pt import BM25
from datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)
nltk.download('punkt', quiet=True)
smooth_fn = SmoothingFunction().method2

# Commit Bench
ds = load_dataset("Maxscha/commitbench")
CommitBenchdiffs = [sample['diff'] for sample in ds['test']]
CommitBenchcommit_messages = [sample['message'] for sample in ds['test']]
bm25 = BM25(device='cuda')
bm25.index(CommitBenchdiffs) 

# Load the data
with open("msgtextV12.json") as f1:
    msgdata = json.load(f1)
with open("difftextV12.json") as f2:
    diffdata = json.load(f2)

# Get the last 7661 items from the dataset
msgdata = msgdata[-7661:]
diffdata = diffdata[-7661:]

tokenized_queries = [data.split() for data in diffdata]

# Initialize LLM
llm = Ollama(model="codellama", base_url="http://localhost:11434")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You will receive a pair of code diff and its corresponding\
        commit message as an exemplar, and a given\
        code diff. Your task is to write a concise commit message\
        according the given code diff under the guidance of the\
        exemplar. Your output should only be the commit message\
        with no other information."),
    ("user", "Code Diff: {retrieved_diff}\
        Commit Message: {retrieved_msg}\
        Code Diff: {query_diff} Commit Message:")
])
chain = prompt | llm

results_lock = threading.Lock()
all_references = []
all_candidates = []

def task1(start_index, end_index):
    local_references = []
    local_candidates = []
    for index in tqdm(range(start_index, end_index), leave=False):
        query = [diffdata[index]]
        doc_scores = bm25.score_batch(query)
        doc_scores = doc_scores.cpu().numpy()[0]
        best_index = np.argmax(doc_scores)
        exemplar_diff = CommitBenchdiffs[best_index]
        exemplar_msg = CommitBenchcommit_messages[best_index]
        # Get response from LLM
        rsp = chain.invoke({
            "retrieved_diff": exemplar_diff,  # exemplar 的 code diff
            "retrieved_msg": exemplar_msg,    # exemplar 的 commit message
            "query_diff": diffdata[index]     # 你想要生成 commit message 的 code diff
        })
        reference = [nltk.word_tokenize(msgdata[index].lower())]
        candidate = nltk.word_tokenize(rsp.lower())
        
        local_references.append(reference)
        local_candidates.append(candidate)
    
    # Safely update the global lists
    with results_lock:
        all_references.extend(local_references)
        all_candidates.extend(local_candidates)

if __name__ == "__main__":
    print("ID of process running main program: {}".format(os.getpid()))
    print("Main thread name: {}".format(threading.current_thread().name))
    print(diffdata[0])
    
    # Calculate the new data size and chunk size
    data_size = len(msgdata)
    chunk_size = data_size // 2
    print(f"Total data size: {data_size}")

    # Create a thread pool with 20 threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tasks to the thread pool
        futures = []
        for i in range(2):
            start = i * chunk_size
            end = start + chunk_size if i < 1 else data_size
            futures.append(executor.submit(task1, start, end))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()


    # Calculate corpus BLEU score
    corpus_bleu_score = corpus_bleu(all_references, all_candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    print(f" Corpus BLEU-4 score with smoothing: {corpus_bleu_score}")



