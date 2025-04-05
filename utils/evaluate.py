from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import math
import multiprocessing
import voyager
from numpy import log2

#################### KUAIREC ####################

def kuairec_eval(user_emb, item_emb, val_data, test_data, top_k, is_validation=True, progress_bar=True, return_per_user=False):
    """
    Returns:
        A dict containing overall Recall@k, nDCG@k, and Coverage@k averages.
        If return_per_user=True, also returns (recall_list, ndcg_list) for significance testing.
    """
    video_emb = item_emb
    neigh = NearestNeighbors(n_neighbors=max(top_k), metric='cosine')
    neigh.fit(video_emb)

    idcg_cache = {}
    def get_idcg(r_size, k):
        cut = min(k, r_size)
        if (cut, k) not in idcg_cache:
            val = 0.0
            for i in range(cut):
                val += 1.0 / np.log2(i + 2)
            idcg_cache[(cut, k)] = val
        return idcg_cache[(cut, k)]

    total_recall = {k: 0.0 for k in top_k}
    total_ndcg = {k: 0.0 for k in top_k}
    user_count = 0
    # Track unique recommended items for each k
    recommended_items = {k: set() for k in top_k}
    total_items = len(video_emb)  # Total items in catalog

    recall_list = {f'Recall@{k}': [] for k in top_k}
    ndcg_list = {f'NDCG@{k}': [] for k in top_k}

    relevant_dict = val_data if is_validation else test_data
    other_dict = test_data if is_validation else val_data

    for u_idx in (tqdm(relevant_dict.keys()) if progress_bar else relevant_dict.keys()):
        relevant_set = relevant_dict[u_idx]
        other_set = other_dict.get(u_idx, set())
        if len(relevant_set) == 0:
            continue

        user_vec = user_emb[u_idx].reshape(1, -1)
        distances, indices = neigh.kneighbors(user_vec, n_neighbors=top_k[-1])
        indices = indices.flatten()

        ranked_items = []
        for vid in indices:
            if vid in other_set:
                continue
            ranked_items.append(vid)
            if len(ranked_items) == max(top_k):
                break

        hits_positions = []
        for rank, vid in enumerate(ranked_items):
            if vid in relevant_set:
                hits_positions.append(rank)

        for k in top_k:
            hits_count = sum(1 for pos in hits_positions if pos < k)
            recall_k = hits_count / float(len(relevant_set))
            total_recall[k] += recall_k
            recall_list[f'Recall@{k}'].append(recall_k)

            dcg_val = sum(1.0 / np.log2(pos + 2) for pos in hits_positions if pos < k)
            idcg_val = get_idcg(len(relevant_set), k)
            ndcg_k = (dcg_val / idcg_val) if idcg_val > 0 else 0.0
            total_ndcg[k] += ndcg_k
            ndcg_list[f'NDCG@{k}'].append(ndcg_k)

            # Add top-k ranked items to coverage set
            recommended_items[k].update(ranked_items[:k])

        user_count += 1

    if user_count == 0:
        return {f'Recall@{k}': 0.0 for k in top_k} | {f'NDCG@{k}': 0.0 for k in top_k} | {f'Coverage@{k}': 0.0 for k in top_k}

    avg_hr = {f'Recall@{k}': total_recall[k] / user_count for k in top_k}
    avg_ndcg = {f'NDCG@{k}': total_ndcg[k] / user_count for k in top_k}
    coverage = {f'Coverage@{k}': len(recommended_items[k]) / total_items for k in top_k}

    if return_per_user:
        return (recall_list | ndcg_list)
    else:
        return (avg_hr | avg_ndcg | coverage)

#################### SOUNDCLOUD ####################
    
def build_voyager_index(track_emb, batch_size=5_000_000, ef_construction=90):
    """
    Build a Voyager index from track_emb in batches.
    :param track_emb: np.ndarray of shape [num_tracks, dim].
    :param batch_size: Number of embeddings to add per batch.
    :param ef_construction: Voyager parameter controlling index build speed vs. quality.
    """
    num_tracks, dim = track_emb.shape

    # Create a Voyager index
    index = voyager.Index(
        space=voyager.Space.Cosine, #voyager.Space.Euclidean,
        num_dimensions=dim,
        max_elements=num_tracks,
        ef_construction=ef_construction
    )
    
    for i in range(0, num_tracks, batch_size):
        batch = track_emb[i : i + batch_size]
        index.add_items(batch)
        print('batch')
    
    return index

# -----------------------------
# Global variables for multiprocessing
# -----------------------------
GLOBAL_USER_EMB = None           # np.array: [num_users, dim]
GLOBAL_VOYAGER = None            # Voyager index (already built)
GLOBAL_VAL_DATA = None           # dict: user_id -> set of relevant item indices
GLOBAL_TEST_DATA = None          # dict: user_id -> set of relevant item indices
GLOBAL_TOPK_LIST = None          # A list of top-K values, e.g. [1, 5, 10, 20]
GLOBAL_IS_VALIDATION = True
GLOBAL_TOPK_MAX = None           # max of top_k list

def _init_globals(user_emb, voyager_index, val_data, test_data,
                  top_k_list, is_validation, top_k_max):
    """
    Initializer for each child process: sets globals for shared data (via fork).
    """
    global GLOBAL_USER_EMB, GLOBAL_VOYAGER, GLOBAL_VAL_DATA
    global GLOBAL_TEST_DATA, GLOBAL_TOPK_LIST, GLOBAL_IS_VALIDATION
    global GLOBAL_TOPK_MAX

    GLOBAL_USER_EMB = user_emb
    GLOBAL_VOYAGER = voyager_index
    GLOBAL_VAL_DATA = val_data
    GLOBAL_TEST_DATA = test_data
    GLOBAL_TOPK_LIST = top_k_list
    GLOBAL_IS_VALIDATION = is_validation
    GLOBAL_TOPK_MAX = top_k_max

def _eval_worker_batch(user_list):
    user_emb = GLOBAL_USER_EMB
    voyager_index = GLOBAL_VOYAGER
    val_data = GLOBAL_VAL_DATA
    test_data = GLOBAL_TEST_DATA
    top_k_list = GLOBAL_TOPK_LIST
    top_k_max = GLOBAL_TOPK_MAX
    is_validation = GLOBAL_IS_VALIDATION

    relevant_dict = val_data if is_validation else test_data
    other_dict = test_data if is_validation else val_data

    batch_queries = np.stack([user_emb[u] for u in user_list])
    neighbors_batch, _ = voyager_index.query(batch_queries, k=top_k_max)

    sum_recall_dict = {k: 0.0 for k in top_k_list}
    sum_ndcg_dict = {k: 0.0 for k in top_k_list}
    recommended_items = {k: set() for k in top_k_list}  # Track unique items per k
    count = 0

    idcg_cache = {}
    def get_idcg(r_size, k):
        cut = min(k, r_size)
        if (cut, k) not in idcg_cache:
            s = 0.0
            for i in range(cut):
                s += 1.0 / log2(i + 2)
            idcg_cache[(cut, k)] = s
        return idcg_cache[(cut, k)]

    for i, u in enumerate(user_list):
        relevant_set = relevant_dict.get(u, set())
        if not relevant_set:
            continue
        other_set = other_dict.get(u, set())

        nn_indices = neighbors_batch[i].tolist()
        ranked_items = []
        for idx in nn_indices:
            if idx not in other_set:
                ranked_items.append(idx)
            if len(ranked_items) == top_k_max:
                break

        hits_positions = [rank for rank, vid in enumerate(ranked_items) if vid in relevant_set]
        r_size = len(relevant_set)
        for k in top_k_list:
            hits_within_k = sum(1 for pos in hits_positions if pos < k)
            recall = hits_within_k / float(r_size)
            sum_recall_dict[k] += recall

            dcg = sum(1.0 / log2(pos + 2) for pos in hits_positions if pos < k)
            idcg = get_idcg(r_size, k)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            sum_ndcg_dict[k] += ndcg

            recommended_items[k].update(ranked_items[:k])
        count += 1

    return (sum_recall_dict, sum_ndcg_dict, recommended_items, count)

def evaluate_user_reco_approx_multiproc(user_emb, voyager_index, val_data, test_data,
                                        top_k_list=[100], is_validation=True, n_proc=64,
                                        progress_bar=True):
    top_k_max = max(top_k_list)
    relevant_dict = val_data if is_validation else test_data
    user_ids = list(relevant_dict.keys())

    total_users = len(user_ids)
    chunk_size = max(1, total_users // n_proc)
    chunks = [user_ids[i:i + chunk_size] for i in range(0, total_users, chunk_size)]

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        processes=n_proc,
        initializer=_init_globals,
        initargs=(user_emb, voyager_index, val_data, test_data, top_k_list, is_validation, top_k_max)
    ) as pool:
        results = list(tqdm(pool.imap(_eval_worker_batch, chunks), total=len(chunks),
                            desc="Evaluating users") if progress_bar else pool.imap(_eval_worker_batch, chunks))

    sum_recall_dict = {k: 0.0 for k in top_k_list}
    sum_ndcg_dict = {k: 0.0 for k in top_k_list}
    recommended_items = {k: set() for k in top_k_list}
    user_count = 0

    for (partial_recall_dict, partial_ndcg_dict, partial_items, cnt) in results:
        for k in top_k_list:
            sum_recall_dict[k] += partial_recall_dict[k]
            sum_ndcg_dict[k] += partial_ndcg_dict[k]
            recommended_items[k].update(partial_items[k])
        user_count += cnt

    if user_count == 0:
        return {f"Recall@{k}": 0.0 for k in top_k_list} | {f"NDCG@{k}": 0.0 for k in top_k_list} | {f"Coverage@{k}": 0.0 for k in top_k_list}

    total_items = voyager_index.num_elements  # Total tracks in the index
    recall_avg = {f"Recall@{k}": (sum_recall_dict[k] / user_count) for k in top_k_list}
    ndcg_avg = {f"NDCG@{k}": (sum_ndcg_dict[k] / user_count) for k in top_k_list}
    coverage = {f"Coverage@{k}": len(recommended_items[k]) / total_items for k in top_k_list}

    return recall_avg | ndcg_avg | coverage

def soundcloud_eval(user_emb, item_emb, val_data, test_data, top_k=[100], is_validation=True, progress_bar=True):
    track_emb = item_emb
    index = build_voyager_index(track_emb)
    del track_emb
    print("Index built")

    metrics = evaluate_user_reco_approx_multiproc(
        user_emb=user_emb,
        voyager_index=index,
        val_data=val_data,
        test_data=test_data,
        top_k_list=top_k,
        is_validation=is_validation,
        n_proc=32,
        progress_bar=progress_bar
    )
    del user_emb
    return metrics