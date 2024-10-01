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

warnings.filterwarnings("ignore", category=UserWarning)
nltk.download('punkt', quiet=True)
smooth_fn = SmoothingFunction().method2


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
llm = Ollama(model="mistral", base_url="http://localhost:11434")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your task is to write a concise commit message\
        according a given code di . Your output should only be\
        the commit message with no other information."),
    ("user", "Code Diff: {query_diff} Commit Message:")
])
chain = prompt | llm

results_lock = threading.Lock()
all_references = []
all_candidates = []

def task1(start_index, end_index):
    local_references = []
    local_candidates = []
    for index in tqdm(range(start_index, end_index), leave=False):
        # Get response from LLM
        rsp = chain.invoke({
            "query_diff": diffdata[index]
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
    #print(diffdata[0])
    
    # Calculate the new data size and chunk size
    data_size = len(msgdata)
    chunk_size = data_size // 5
    print(f"Total data size: {data_size}")

    # Create a thread pool with 20 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
    # Submit tasks to the thread pool
        futures = []

        for i in range(5):
            start = i * chunk_size
            end = start + chunk_size if i < 4 else data_size
            futures.append(executor.submit(task1, start, end))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    # Calculate corpus BLEU score
    corpus_bleu_score = corpus_bleu(all_references, all_candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    print(f" Corpus BLEU-4 score with smoothing: {corpus_bleu_score}")



