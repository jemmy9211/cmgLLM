import json
from tqdm.auto import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer, util  # type: ignore
from datasets import load_dataset
import torch

# Import TF-IDF Vectorizer and cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)
nltk.download('punkt', quiet=True)
smooth_fn = SmoothingFunction().method2

modelname = "llama3.2"

# Load CommitBench dataset
ds = load_dataset("Maxscha/commitbench")
CommitBenchdiffs = [sample['diff'] for sample in ds['test']]
CommitBenchcommit_messages = [sample['message'] for sample in ds['test']]

# Initialize Sentence Transformer model for semantic retrieval
st_model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = st_model.encode(CommitBenchdiffs, convert_to_tensor=True)

# Initialize TF-IDF Vectorizer for lexical retrieval
tfidf_vectorizer = TfidfVectorizer()
corpus_tfidf = tfidf_vectorizer.fit_transform(CommitBenchdiffs)

# Load your data
with open("msgtextV12.json") as f1:
    msgdata = json.load(f1)
with open("difftextV12.json") as f2:
    diffdata = json.load(f2)

msgdata = msgdata[:20000]
diffdata = diffdata[:20000]

# Ensure data lengths match
assert len(msgdata) == len(diffdata), "Mismatch between message and diff data lengths."

# Initialize LLM
llm = Ollama(model=modelname, base_url="http://localhost:11434")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You will receive a pair of code diff and its corresponding\
        commit message as an exemplar, and a given\
        code diff. Your task is to write a concise commit message\
        according to the given code diff under the guidance of the\
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

def retrieve_exemplar(query_diff):
    # Semantic similarity
    query_embedding = st_model.encode(query_diff, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    # TF-IDF similarity
    query_tfidf = tfidf_vectorizer.transform([query_diff])
    tfidf_scores = cosine_similarity(query_tfidf, corpus_tfidf)[0]

    # Combine the similarities (weights can be adjusted)
    combined_scores = 0.5 * semantic_scores + 0.5 * tfidf_scores

    best_index = np.argmax(combined_scores)
    return CommitBenchdiffs[best_index], CommitBenchcommit_messages[best_index]

def task1(indices, progress_bar):
    local_references = []
    local_candidates = []
    for index in indices:
        try:
            query_diff = diffdata[index]
            exemplar_diff, exemplar_msg = retrieve_exemplar(query_diff)

            # Get response from LLM
            rsp = chain.invoke({
                "retrieved_diff": exemplar_diff,
                "retrieved_msg": exemplar_msg,
                "query_diff": query_diff
            })

            reference = [nltk.word_tokenize(msgdata[index].lower())]
            candidate = nltk.word_tokenize(rsp.lower())

            local_references.append(reference)
            local_candidates.append(candidate)
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            continue
        finally:
            progress_bar.update(1)  # Update the progress bar

    with results_lock:
        all_references.extend(local_references)
        all_candidates.extend(local_candidates)

if __name__ == "__main__":
    print(f"Method: Hybrid Retrieval (Semantic + TF-IDF)")
    print(f"Model: {modelname}")

    data_size = len(msgdata)
    print(f"Total data size: {data_size}")

    num_threads = 1  # Adjust as needed
    chunk_size = data_size // num_threads
    indices_list = []

    # Distribute chunks and account for the remainder
    for i in range(num_threads):
        start_index = i * chunk_size
        # The last chunk should include the remainder
        end_index = (i + 1) * chunk_size if i != num_threads - 1 else data_size
        indices_list.append(range(start_index, end_index))

    # Initialize the progress bar
    with tqdm(total=data_size, desc="Processing", unit="tasks") as progress_bar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(task1, indices, progress_bar) for indices in indices_list]
            for future in as_completed(futures):
                pass

    # Calculate corpus BLEU score
    corpus_bleu_score = corpus_bleu(
        all_references,
        all_candidates,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth_fn
    )
    print(f"Corpus BLEU-4 score with smoothing: {corpus_bleu_score}")
