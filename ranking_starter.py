#!/usr/bin/env python3
"""
Evaluate retrieval quality on the RAG sample dataset.

Computes Precision@K, Recall@K, and nDCG@K
for each query and averages the results.

Assumes the dataset columns:
  query_id, query_text, candidate_id, candidate_text,
  baseline_rank, baseline_score, gold_label
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("rag_sample_queries_candidates.csv")

# Ensure results are ordered by the baseline rank
df.sort_values(["query_id", "baseline_rank"], inplace=True)
df2 = pd.read_csv("LLM_Output3.csv")
df2.sort_values(["LLM_Values"], inplace=True)

# print(df[["query_id","baseline_score"]])
# print(df2[["query_id","LLM_Values"]])

# # print(df.columns)
# # print(df["candidate_text"])
# temp = df.groupby("query_id")
# # print(temp.ngroups)    
# # n = temp.ngroups      # number of groups
# # print(temp.groups.keys()) 
# # print(temp.get_group(1).head())

# all_q = []
# for qid,group in temp:
#     temp_q = []
#     # print(qid)
#     for idx, row in group.iterrows():
#         # print(row)
#         query =  str(row["query_id"]) + " " +  str(row["query_text"] + ": \n  Canadate text: " + str(row["candidate_text"]))
#         temp_q.append(query)
#     # print(temp_q)
#     all_q.append(temp_q)
# # print(all_q)
# for i in all_q:
#     print(i)
#     for l in i:
#         # print(b)
#         b = l.split(":")
#         print(b)
        # ---------------------------------------------------------------------
# 2. Metric helpers
# ---------------------------------------------------------------------
def precision_at_k(labels, k):
    """labels: list/array of 0/1 relevance sorted by baseline rank"""
    topk = labels[:k]
    return np.sum(topk) / len(topk)

def recall_at_k(labels, k):
    """Recall = retrieved relevant / total relevant"""
    total_relevant = np.sum(labels)
    if total_relevant == 0:
        return np.nan  # undefined
    topk = labels[:k]
    return np.sum(topk) / total_relevant

def ndcg_at_k(labels, k):
    """Compute nDCG@k with binary relevance (0/1)."""
    labels = np.array(labels)
    k = min(k, len(labels))
    gains = (2 ** labels[:k] - 1)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)

    # Ideal DCG: sorted by true relevance
    ideal = np.sort(labels)[::-1]
    ideal_gains = (2 ** ideal[:k] - 1)
    idcg = np.sum(ideal_gains * discounts)
    return 0.0 if idcg == 0 else dcg / idcg

# def ndcg_score(one, two):
#     labels = np.array(one)

# ---------------------------------------------------------------------
# 3. Compute metrics per query
# ---------------------------------------------------------------------
results = []
K = 3

LLM_Results = []

for qid, group in df.groupby("query_id"):
    labels = group["gold_label"].tolist()
    p = precision_at_k(labels, K)
    r = recall_at_k(labels, K)
    n = ndcg_at_k(labels, K)
    results.append({"query_id": qid, f"precision@{K}": p, f"recall@{K}": r, f"nDCG@{K}": n})



############################################
# Comparing based off LMM_values
############################################
for qid, group in df2.groupby("query_id"):
    labels = group["gold_label"].tolist()
    p = precision_at_k(labels, K)
    r = recall_at_k(labels, K)
    n = ndcg_at_k(labels, K)
    LLM_Results.append({"LLM_Values": qid, f"precision@{K}": p, f"recall@{K}": r, f"nDCG@{K}": n})

# for qid, group in df2.groupby("query_id"):
#     y_true = group.sort_values("baseline_rank")["gold_label"].to_numpy()
#     y_pred = group.sort_values("baseline_rank")["baseline_score"].to_numpy()
#     # baseline_ndcg = ndcg_score([y_true], [y_pred])
#     y_pred_llm = group.sort_values("LLM_Values", ascending=False)["LLM_Values"].to_numpy()
#     print(y_true)
#     # ndcg_llm = ndcg_score([y_true], [y_pred_llm])
#     # print(f"Query {qid}: baseline nDCG={baseline_ndcg:.3f}, LLM nDCG={ndcg_llm:.3f}")


metrics = pd.DataFrame(results)

# ---------------------------------------------------------------------
# 4. Display per-query and average metrics
# ---------------------------------------------------------------------
print(metrics.round(3))
print("\nAverage metrics:")
print(metrics[[f"precision@{K}", f"recall@{K}", f"nDCG@{K}"]].mean().round(3))


metrics2 = pd.DataFrame(LLM_Results)


# ---------------------------------------------------------------------
# 5. Display per-query and average metrics2
# ---------------------------------------------------------------------
print(metrics2.round(3))
print("\nAverage metrics:")
print(metrics2[[f"precision@{K}", f"recall@{K}", f"nDCG@{K}"]].mean().round(3))
