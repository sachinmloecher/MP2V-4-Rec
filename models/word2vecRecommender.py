from collections import defaultdict
from models.baseRecommender import Recommender
from utils.evaluate import kuairec_eval, soundcloud_eval
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import multiprocessing
import torch


class Word2VecRecommender(Recommender):
    def __init__(self, config):
        super().__init__(config)
    
    def fit(self, sequences, train_interactions, val_data, test_data):
        self.evaluate = kuairec_eval
        unique_users = train_interactions['user_id'].unique()
        unique_videos = train_interactions['video_id'].unique()
        num_users = len(unique_users)
        num_videos = len(unique_videos)
        train_interactions = train_interactions[train_interactions['watch_ratio'] >= 2].sort_values(['user_id', 'time'])


        user2idx = {uid: i for i, uid in enumerate(unique_users)}
        video2idx = {vid: i for i, vid in enumerate(unique_videos)}

        model = Word2Vec(
                    vector_size=self.embedding_dim,
                    window=self.window,
                    workers=multiprocessing.cpu_count(),
                    sg=1,
                    min_count=1,
                    compute_loss=True,
                )
        model.build_vocab(sequences)
        model.train(
                    corpus_iterable=sequences,
                    total_examples=len(sequences),
                    epochs=self.epochs,
                )
        model_vocab = list(model.wv.index_to_key)

        video_emb = np.zeros((num_videos, self.embedding_dim), dtype=np.float32)
        for vid, idx in video2idx.items():
            if str(vid) in model_vocab:
                video_emb[idx] = model.wv[str(vid)]

        video_emb = torch.tensor(video_emb, dtype=torch.float32)

        user_to_videos = defaultdict(list)
        for _, row in train_interactions.iterrows():
            user_id = row['user_id']
            video_id_str = str(row['video_id'])  # Convert to string for Word2Vec keys
            user_to_videos[user_id].append(video_id_str)

        # User embeddings are their average video embeddings from training watch history
        user_emb = torch.zeros((num_users, self.embedding_dim), dtype=torch.float32)
        for uid, u_idx in user2idx.items():
            vids_watched = user_to_videos[uid]
            if not vids_watched:
                continue  # user_emb remains zero if no videos

            sum_vec = np.zeros(self.embedding_dim, dtype=np.float32)
            count = 0
            for vid_str in vids_watched:
                if vid_str in model.wv:  # If the video ID is in the W2V vocab
                    sum_vec += model.wv[vid_str]
                    count += 1
            if count > 0:
                user_emb[u_idx] = torch.tensor(sum_vec / count, dtype=torch.float32)
                
        # Evaluate W2V embeddings
        val_metrics = self.evaluate(
            user_emb=user_emb,
            item_emb=video_emb,
            val_data=val_data,
            test_data=test_data,
            is_validation=True,
            top_k=self.k,
            progress_bar=False
        )
        self.user_emb = user_emb
        self.video_emb = video_emb
        return {'Epoch': self.epochs, 'val_metrics': val_metrics}