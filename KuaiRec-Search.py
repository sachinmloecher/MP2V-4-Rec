# Imports
import logging
import os
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
import torch
import json
import multiprocessing
from collections import defaultdict

from models.metapath2vecRecommender import MetaPath2VecRecommender
from models.Wmetapath2vecRecommender import WMetaPath2VecRecommender
logging.getLogger("gensim").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

KUAIREC_ADJ_LIST = 'KuaiRec/adj_data.pkl'

# Global search parameters
MODELS = ['metapath2vec', 'Wmetapath2vec']
METAPATH = [
    ('user', 'follows', 'user'),
    ('user', 'watches', 'video'),
    ('video', 'watched_by', 'user')
]
WALK_LENGTHS = [4, 10, 15, 25]
CONTEXT_SIZES = [2, 3, 6, 10, 15]
WALKS_PER_NODE = [3, 5, 10, 15, 30]
NUM_NEG_SAMPLES = [3, 6, 10]
LR = 0.0125
MAX_EPOCHS = 60
PATIENCE = 16
BATCH_SIZE = 300
Ks = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EMBEDDING_DIM = 50

def init_logger(log_dir):
    """Initialize a logger that writes to a unique log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"output.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear previous handlers if reusing the same script
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # Add handlers
    logger.addHandler(file_handler)
    return logger

def w2v_emb():
    df_watch = pd.read_csv('KuaiRec/big_matrix.csv')
    df_watch = df_watch[df_watch['watch_ratio'] >= 2].sort_values(['user_id', 'time'])

    unique_users = df_watch['user_id'].unique()
    unique_videos = df_watch['video_id'].unique()

    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    video2idx = {vid: i for i, vid in enumerate(unique_videos)}
    
    train = pd.read_parquet('KuaiRec/train_sequences.parquet')
    sequences = train['video_id'].apply(lambda x: list(map(str, x))).tolist()
    model = Word2Vec(
                vector_size=EMBEDDING_DIM,
                window=15,
                workers=multiprocessing.cpu_count(),
                sg=1,
                min_count=1,
                compute_loss=True,
            )
    model.build_vocab(sequences)
    model.train(
                corpus_iterable=sequences,
                total_examples=len(sequences),
                epochs=75,
            )
    model_vocab = list(model.wv.index_to_key)
    with open(KUAIREC_ADJ_LIST, "rb") as f:
        adj_data = pickle.load(f)
        
    video_emb = np.zeros((adj_data['num_nodes_dict']['video'], EMBEDDING_DIM), dtype=np.float32)
    for vid, idx in video2idx.items():
        if str(vid) in model_vocab:
            video_emb[idx] = model.wv[str(vid)]

    video_emb = torch.tensor(video_emb, dtype=torch.float32)
    
    num_users = adj_data['num_nodes_dict']['user']

    user_to_videos = defaultdict(list)
    for _, row in df_watch.iterrows():
        user_id = row['user_id']
        video_id_str = str(row['video_id'])  # Convert to string for Word2Vec keys
        user_to_videos[user_id].append(video_id_str)

    # Initialize user embeddings to their average video embeddings from training watch history
    user_emb = torch.zeros((num_users, EMBEDDING_DIM), dtype=torch.float32)
    for uid, u_idx in user2idx.items():
        vids_watched = user_to_videos[uid]
        if not vids_watched:
            continue  # user_emb remains zero if no videos

        sum_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        count = 0
        for vid_str in vids_watched:
            if vid_str in model.wv:  # If the video ID is in the W2V vocab
                sum_vec += model.wv[vid_str]
                count += 1
        if count > 0:
            user_emb[u_idx] = torch.tensor(sum_vec / count, dtype=torch.float32)
    return (video_emb, user_emb)


def main():
    # Load dataset
    with open(KUAIREC_ADJ_LIST, "rb") as f:
        adj_data = pickle.load(f)
    with open('KuaiRec/val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('KuaiRec/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    # Generate precomputed W2V embeddings (users, videos)
    w2v_embeddings = w2v_emb()
    # Hyperparam search
    total_configs = (
        len(MODELS) * len(WALK_LENGTHS) * len(CONTEXT_SIZES) * len(WALKS_PER_NODE) * len(NUM_NEG_SAMPLES)
    )
    with tqdm(total=total_configs, desc="Hyperparameter Search Progress") as pbar:
        for m in MODELS:
            for walk_length in WALK_LENGTHS:
                for context_size in CONTEXT_SIZES:
                    for walks_per_node in WALKS_PER_NODE:
                        for num_negative_samples in NUM_NEG_SAMPLES:
                            config = {
                                    'walk_length': walk_length,
                                    'context_size': context_size,
                                    'walks_per_node': walks_per_node,
                                    'num_negative_samples': num_negative_samples,
                                    'lr': LR,
                                    'embedding_dim': EMBEDDING_DIM,
                                    'max_epochs': MAX_EPOCHS,
                                    'patience': PATIENCE,
                                    'batch_size': BATCH_SIZE,
                                    'k': Ks,
                                    'eval_freq': 1,
                                    'device': 'cuda',
                                    'verbose': True,
                                    'num_workers': 32,
                                    'dataset': 'KuaiRec'
                            }
                            run_dir = f"runs/KuaiRec/{m}_wl{walk_length}_cs{context_size}_wpn{walks_per_node}_nns{num_negative_samples}"
                            logger = init_logger(run_dir)
                            config_file = f"{run_dir}/config.json"
                            with open(config_file, "w") as f:
                                json.dump(config, f)
                            try:
                                if m == 'metapath2vec':
                                    model = MetaPath2VecRecommender(config)
                                elif m == 'Wmetapath2vec':
                                    model = WMetaPath2VecRecommender(config)
                                else:
                                    raise ValueError(f"Unknown model: {m}")

                                results = model.fit(adj_data, w2v_embeddings, METAPATH, val_data, test_data, logger)
                            except Exception as e:
                                logger.error(f"Error creating model: {e}")
                                pbar.update(1)
                                continue
                            # Save results to file
                            results_file = f"{run_dir}/results.txt"
                            with open(results_file, "w") as f:
                                json.dump(results, f)
                            del model
                            
                            logger.info(f"Completed experiment: {run_dir}")
                            logger.info(f"Results saved to {results_file}")
                            pbar.update(1)
                            
if __name__ == "__main__":
    main()
