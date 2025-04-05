import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pickle
import random
import itertools
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import glob
import os
import random
import time

import torch
from torch_geometric.data import HeteroData
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch_geometric.index import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import autocast
import pickle
import time
import os
import numpy as np

EPS = 1e-15

class MetaPath2Vec(nn.Module):
    def __init__(self, precomputed_adj, embedding_dim, metapath, walk_length, context_size, 
                 walks_per_node=1, num_negative_samples=1, sparse=True, device="cuda"):
        super().__init__()
        self.device = device
        self.rowptr_dict = precomputed_adj["rowptr_dict"]
        self.col_dict = precomputed_adj["col_dict"]
        self.rowcount_dict = precomputed_adj["rowcount_dict"]
        self.num_nodes_dict = precomputed_adj["num_nodes_dict"]

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError("Invalid metapath chain.")
        assert walk_length + 1 >= context_size

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        types = sorted(set([x[0] for x in metapath]) | set([x[-1] for x in metapath]))
        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += self.num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # Match original: no immediate .to(device)
        self.embedding = nn.Embedding(count + 1, embedding_dim, sparse=sparse, dtype=torch.bfloat16)
        self.dummy_idx = count
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, node_type: str, batch: Tensor = None) -> Tensor:
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            batch = sample(
                self.rowptr_dict[edge_type],
                self.col_dict[edge_type],
                self.rowcount_dict[edge_type],
                batch,
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
            ).view(-1)
            rws.append(batch)
        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw[rw > self.dummy_idx] = self.dummy_idx
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0),), dtype=torch.long)
            rws.append(batch)
        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

def sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, subset: Tensor,
           num_neighbors: int, dummy_idx: int) -> Tensor:
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]
    rand = torch.rand((subset.size(0), num_neighbors), device=subset.device)
    rand *= count.to(rand.dtype).view(-1, 1)
    rand = rand.to(torch.long) + rowptr[subset].view(-1, 1)
    rand = rand.clamp(max=col.numel() - 1)
    col = col[rand] if col.numel() > 0 else rand
    col[mask | (count == 0)] = dummy_idx
    return col

def train(config):
    device = torch.device("cuda")  # Single GPU
    torch.cuda.empty_cache()
    print(f"GPU memory before start: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Load SoundCloud data
    with open("SoundCloud/adj_data_90_active_sampled.pkl", "rb") as f:
        adj_data = pickle.load(f)
    print("Loaded data")

    # Initialize model on CPU
    model = MetaPath2Vec(
        precomputed_adj=adj_data,
        embedding_dim=config["embedding_dim"],
        metapath=[('user', 'listens', 'track'), ('track', 'listened_by', 'user')], #('user', 'follows', 'user'), 
        walk_length=config["walk_length"],
        context_size=config["context_size"],
        walks_per_node=config["walks_per_node"],
        num_negative_samples=config["num_negative_samples"],
        sparse=True,
        device=device
    )
    print(f"Model created (on CPU), GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Comment out pre-trained embedding initialization for testing
    user_embeddings = torch.tensor(np.load('SoundCloud/user_embeddings_active_sampled.npy'), dtype=torch.bfloat16)
    track_embeddings = torch.tensor(np.load('SoundCloud/track_embeddings_active.npy'), dtype=torch.bfloat16)
    with torch.no_grad():
         model.forward('user').copy_(user_embeddings)
         model.forward('track').copy_(track_embeddings)
    del user_embeddings, track_embeddings
    print(f"Embeddings initialized on CPU")

    # Move model to GPU after initialization
    model.to(device)
    print(f"Model moved to GPU, GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Use SparseAdam optimizer
    #optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config["lr"])
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config["lr"])

    """ user_listens_rowcount = adj_data['rowcount_dict'][('user', 'listens', 'track')]
    active_users = user_listens_rowcount >= 5  # Require ≥5 listens
    active_user_ids = torch.where(active_users)[0]
    print(f"Highly active users (≥5 listens): {active_user_ids.size(0)}/{adj_data['num_nodes_dict']['user']} "
          f"({active_user_ids.size(0) / adj_data['num_nodes_dict']['user'] * 100:.2f}%)")
    print(f"Avg listens per active user: {user_listens_rowcount[active_users].float().mean().item():.2f}")

    track_listened_rowcount = adj_data['rowcount_dict'][('track', 'listened_by', 'user')]
    active_tracks = track_listened_rowcount >= 5  # Require ≥5 listeners
    print(f"Highly active tracks (≥5 listeners): {active_tracks.sum().item()}/{adj_data['num_nodes_dict']['track']} "
          f"({active_tracks.sum().item() / adj_data['num_nodes_dict']['track'] * 100:.2f}%)")
    print(f"Avg listeners per active track: {track_listened_rowcount[active_tracks].float().mean().item():.2f}") """
    del adj_data

    # Create DataLoader with smaller batch size
    loader = DataLoader(
        range(model.num_nodes_dict[model.metapath[0][0]]),
        batch_size=config["batch_size"],
        shuffle=True,  # Enable shuffling for better learning
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=model._sample
    )
    print(f"Created DataLoader with {len(loader)} batches")
    """ # DataLoader with only active users
    loader = DataLoader(
        active_user_ids,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=model._sample
    )
    print(f"Created DataLoader with {len(loader)} batches")
    isolated_users_mask = (~active_users).to(device)
    isolated_tracks_mask = (~active_tracks).to(device) """

    for epoch in range(1, config["epochs"] + 1):
        total_loss = 0
        epoch_start = time.time()
        model.train()
        for i, (pos_rw, neg_rw) in enumerate(loader):
            start_time = time.time()
        
            pos_rw_gpu = pos_rw.to(device, non_blocking=True)
            neg_rw_gpu = neg_rw.to(device, non_blocking=True)
            if i % 1000 == 0:
                pos_dummy_count = (pos_rw == model.dummy_idx).sum().item()
                neg_dummy_count = (neg_rw == model.dummy_idx).sum().item()
                pos_total_nodes = pos_rw.numel()
                neg_total_nodes = neg_rw.numel()
                pos_dummy_pct = (pos_dummy_count / pos_total_nodes) * 100
                neg_dummy_pct = (neg_dummy_count / neg_total_nodes) * 100
                print(f"Epoch {epoch}, Batch {i}: Pos_rw dummy %: {pos_dummy_pct:.2f}% "
                      f"({pos_dummy_count}/{pos_total_nodes})")
                print(f"Epoch {epoch}, Batch {i}: Neg_rw dummy %: {neg_dummy_pct:.2f}% "
                      f"({neg_dummy_count}/{neg_total_nodes})")

            optimizer.zero_grad()
            with autocast('cuda'):  # Use autocast without dtype for simplicity (bfloat16 handled by embeddings)
                loss = model.loss(pos_rw_gpu, neg_rw_gpu)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    
        print(f"Epoch {epoch} Time: {time.time() - epoch_start:.4f}s, Total Loss: {total_loss:.4f}")
        if epoch % config["eval_freq"] == 0:
            with torch.no_grad():
                track_embeddings = model.forward('track').cpu().to(torch.float32).numpy()
                user_embeddings = model.forward('user').cpu().to(torch.float32).numpy()
            save_path = f'runs/test/wl_{config["walk_length"]}_cs_{config["context_size"]}_wpn_{config["walks_per_node"]}_nns_{config["num_negative_samples"]}_lr_{config["lr"]}/epoch_{epoch}'
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, f't_embeddings7_{epoch}.npy'), track_embeddings)
            np.save(os.path.join(save_path, f'u_embeddings7_{epoch}.npy'), user_embeddings)
            print(f"Saved embeddings for epoch {epoch}")

if __name__ == "__main__":
    config = {
        "embedding_dim": 150,
        "walk_length": 18,
        "context_size": 2,
        "walks_per_node": 30,
        "num_negative_samples": 10,
        "batch_size": 1028,  # Reduced batch size for memory
        "num_workers": 12,  # Adjust based on CPU cores
        "lr": 0.1,
        "epochs": 20,
        "eval_freq": 4,
    }
    
    train(config)