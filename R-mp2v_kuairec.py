import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import pickle
import time
import os
import numpy as np
from utils.evaluate import kuairec_eval

EPS = 1e-15

class WeightedMetaPath2Vec(nn.Module):
    def __init__(
        self,
        precomputed_adj,
        embedding_dim: int,
        metapath,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        sparse: bool = False,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.rowptr_dict = precomputed_adj["rowptr_dict"]
        self.col_dict = precomputed_adj["col_dict"]
        self.rowcount_dict = precomputed_adj["rowcount_dict"]
        self.num_nodes_dict = precomputed_adj["num_nodes_dict"]
        # Check for edge weights; KuaiRec may not have them
        edge_weight_dict = precomputed_adj.get("edge_weight_dict", {})

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError("Invalid metapath chain.")
        assert walk_length + 1 >= context_size, "Walk length must be >= context size."

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        # Handle edge weights (use uniform if not provided)
        self.weight_dict = {}
        for k in self.rowptr_dict.keys():
            w = edge_weight_dict.get(k)
            self.weight_dict[k] = w.cpu() if w is not None else torch.ones_like(self.col_dict[k], dtype=torch.float)

        # Compute prefix sums for weighted sampling
        self.prefix_sum_dict = {}
        for keys in self.rowptr_dict.keys():
            self.prefix_sum_dict[keys] = self.compute_prefix_sums(self.weight_dict[keys])

        types = {x[0] for x in metapath} | {x[-1] for x in metapath}
        types = sorted(list(types))
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

        self.embedding = nn.Embedding(count + 1, embedding_dim, sparse=sparse, dtype=torch.bfloat16)
        self.dummy_idx = count
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def compute_prefix_sums(self, weight: Tensor) -> Tensor:
        return weight.cumsum(dim=0)

    def forward(self, node_type: str, batch: Tensor = None) -> Tensor:
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            rowptr = self.rowptr_dict[edge_type]
            col = self.col_dict[edge_type]
            prefix_sum = self.prefix_sum_dict[edge_type]
            batch = weighted_sample(
                rowptr,
                col,
                prefix_sum,
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

def weighted_sample(rowptr, col, prefix_sum, subset, num_neighbors, dummy_idx):
    assert num_neighbors == 1, "This implementation supports num_neighbors=1 only"
    device = subset.device
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    start = rowptr[subset]
    end = rowptr[subset + 1]
    total_weights = prefix_sum[end - 1] - torch.where(
        start > 0, prefix_sum[start - 1], torch.tensor(0.0, device=device)
    )
    rand = torch.rand(subset.size(0), device=device) * total_weights
    targets = rand + torch.where(
        start > 0, prefix_sum[start - 1], torch.tensor(0.0, device=device)
    )
    global_indices = torch.searchsorted(prefix_sum, targets, right=True)
    global_indices = torch.min(torch.max(global_indices, start), end - 1)
    global_indices = torch.where(total_weights > 0, global_indices, torch.tensor(0, device=device))
    col_indices = col[global_indices]
    col_indices[mask | (total_weights == 0)] = dummy_idx
    return col_indices

def train(config):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    print(f"GPU memory before start: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Load KuaiRec data
    with open("KuaiRec/adj_data.pkl", "rb") as f:
        adj_data = pickle.load(f)
    print("Loaded KuaiRec data")
    with open("KuaiRec/val_data.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open("KuaiRec/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    # Initialize model on CPU
    model = WeightedMetaPath2Vec(
        precomputed_adj=adj_data,
        embedding_dim=config["embedding_dim"],
        metapath=[('user', 'follows', 'user'), ('user', 'watches', 'video'), ('video', 'watched_by', 'user')],
        walk_length=config["walk_length"],
        context_size=config["context_size"],
        walks_per_node=config["walks_per_node"],
        num_negative_samples=config["num_negative_samples"],
        sparse=True,
        device=device
    )
    print(f"Model created (on CPU), GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Move model to GPU
    model.to(device)
    print(f"Model moved to GPU, GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config["lr"])

    # Create DataLoader
    loader = DataLoader(
        range(model.num_nodes_dict[model.metapath[0][0]]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=model._sample
    )
    print(f"Created DataLoader with {len(loader)} batches")

    for epoch in range(1, config["epochs"] + 1):
        total_loss = 0
        epoch_start = time.time()
        model.train()
        for i, (pos_rw, neg_rw) in enumerate(loader):
            pos_rw_gpu = pos_rw.to(device, non_blocking=True)
            neg_rw_gpu = neg_rw.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model.loss(pos_rw_gpu, neg_rw_gpu)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Time: {time.time() - epoch_start:.4f}s, Total Loss: {total_loss:.4f}")
        if epoch % config["eval_freq"] == 0:
            with torch.no_grad():
                model.eval()
                user_emb = model.forward('user').cpu().to(torch.float32).numpy()
                video_emb = model.forward('video').cpu().to(torch.float32).numpy()
            val_metrics = kuairec_eval(
                user_emb=user_emb,
                item_emb=video_emb,
                val_data=val_data,
                test_data=test_data,
                is_validation=True,
                top_k=[100],
                progress_bar=False
            )
            print(f"Validation metrics: {val_metrics}")

            #save_path = f'runs/KuaiRec/wl_{config["walk_length"]}_cs_{config["context_size"]}_wpn_{config["walks_per_node"]}_nns_{config["num_negative_samples"]}_lr_{config["lr"]}/epoch_{epoch}'
            #os.makedirs(save_path, exist_ok=True)
            #np.save(os.path.join(save_path, 'video_embeddings.npy'), video_emb)
            #np.save(os.path.join(save_path, 'user_embeddings.npy'), user_emb)
            #print(f"Saved embeddings for epoch {epoch}")

if __name__ == "__main__":
    config = {
        "embedding_dim": 50,
        "walk_length": 15,
        "context_size": 3,
        "walks_per_node": 25,
        "num_negative_samples": 6,
        "batch_size": 2048,
        "num_workers": 8,
        "lr": 0.2,
        "epochs": 40,
        "eval_freq": 1,
    }
    train(config)