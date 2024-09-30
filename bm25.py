import sqlite3
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# 初始化資料庫
def init_db(db_name='vectors.db'):
    conn = sqlite3.connect(db_name)
    return conn

# 從資料庫查詢所有 diff 和 commit message
def query_diff_from_db(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT id, diff, commit_message FROM vectors')
    results = cursor.fetchall()
    corpus = [(row[0], row[1], row[2]) for row in results]
    return corpus

# 使用BM25進行檢索
def bm25_retrieve(query_diff, corpus_diffs):
    tokenized_corpus = [doc.split() for doc in corpus_diffs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query_diff.split()
    scores = bm25.get_scores(tokenized_query)
    return scores

# 搜索最相关的 diff 和 commit message
def search_relevant_diff(query_diff, conn):
    # 從資料庫查詢 diff 和 commit message
    corpus_data = query_diff_from_db(conn)
    corpus_diffs = [data[1] for data in corpus_data]
    
    # BM25 檢索
    bm25_scores = bm25_retrieve(query_diff, corpus_diffs)

    # 返回最相關的 diff-message pair
    best_match_idx = np.argmax(bm25_scores)
    best_match = corpus_data[best_match_idx]
    return best_match

# 主檢索函數，提供模塊化接口
def bm25_search(query_diff, db_name='vectors.db'):
    # 初始化資料庫
    conn = init_db(db_name)
    
    # 查找最相關的 diff-message pair
    best_match = search_relevant_diff(query_diff, conn)
    
    print(f"Most similar diff: {best_match[1]}")
    print(f"Commit message: {best_match[2]}")
    return best_match

# 使用範例
if __name__ == "__main__":
    query_diff = "diff --git a/camel-core/src/main/java/org/apache/camel/processor/MarshalProcessor.java b/camel-core/src/main/java/org/apache/camel/processor/MarshalProcessor.java index efa2b8e476..7a45624c9b 100644  --- a/camel-core/src/main/java/org/apache/camel/processor/MarshalProcessor.java         +++ b/camel-core/src/main/java/org/apache/camel/processor/MarshalProcessor.java   @@ -82,7 +82,7 @@ public class MarshalProcessor extends ServiceSupport implements AsyncProcessor,   byte[] data = os.toByteArray();  out.setBody(data);  }    -  } catch (Exception e) {  + } catch (Throwable e) {     // remove OUT message, as an exception occurred  exchange.setOut(null); exchange.setException(e); "  # 替換為你要查詢的 diff
    bm25_search(query_diff)
