from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.index import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index

EPS = 1e-15

class WeightedMetaPath2Vec(torch.nn.Module):
    def __init__(
        self,
        precomputed_adj: Dict[str, Dict[EdgeType, torch.Tensor]],  # Precomputed adjacency data
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        sparse: bool = False,
    ):
        super().__init__()
        
        self.rowptr_dict = precomputed_adj["rowptr_dict"]
        self.col_dict = precomputed_adj["col_dict"]
        self.rowcount_dict = precomputed_adj["rowcount_dict"]
        num_nodes_dict = precomputed_adj["num_nodes_dict"]
        edge_weight_dict = precomputed_adj["edge_weight_dict"]
        self.num_nodes_dict = num_nodes_dict

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "Invalid metapath: Destination node type of one relation "
                    "must match the source node type of the next."
                )

        assert walk_length + 1 >= context_size, "Walk length must be >= context size."
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "Walk length is longer than the metapath but the metapath is not a cycle."
            )

        # 5. Store hyperparameters:
        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        # 6. Store edge weights (for weighted sampling):
        #    For each edge type, if no weight is provided, we use ones.
        self.weight_dict = {}
        for k, w in edge_weight_dict.items():
            self.weight_dict[k] = w.cpu() if w is not None else None

        # 7. Build the prefix sum dictionary:
        self.prefix_sum_dict = {}
        for keys in self.rowptr_dict.keys():
            if self.weight_dict[keys] is None:
                # Use uniform weight of 1 if no weight is provided:
                w = torch.ones_like(self.col_dict[keys], dtype=torch.float)
            else:
                w = self.weight_dict[keys]
                # Sanity check: the weight vector must match the number of edges.
                assert w.shape[0] == self.col_dict[keys].shape[0], \
                    f"Weight and col must have same length for edge {keys}"
            # Call the standalone function:
            self.prefix_sum_dict[keys] = self.compute_prefix_sums(w)

        # 8. Build flat index ranges for each node type:
        types = {x[0] for x in metapath} | {x[-1] for x in metapath}
        types = sorted(list(types))
        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        # 9. Build the offset tensor:
        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # 10. Create the embedding layer (plus one extra dummy index for isolated nodes):
        self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        self.dummy_idx = count

        # 11. Reset parameters (initialize embeddings, etc.)
        self.reset_parameters()


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def compute_prefix_sums(self, weight: Tensor) -> Tensor:
        r"""
        Returns a 1D prefix sum array of shape [col.size(0)], 
        where prefix[i] = sum of weights up to index i.
        """
        prefix = weight.cumsum(dim=0)
        return prefix

    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type`.
        """
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph.

        Args:
            **kwargs (optional): Arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self._sample, **kwargs)

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
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z: Tensor, train_y: Tensor, test_z: Tensor,
             test_y: Tensor, solver: str = "lbfgs", *args, **kwargs) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.embedding.weight.size(0) - 1}, '
                f'{self.embedding.weight.size(1)})')


def sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, subset: Tensor,
           num_neighbors: int, dummy_idx: int) -> Tensor:

    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = torch.rand((subset.size(0), num_neighbors), device=subset.device)
    rand *= count.to(rand.dtype).view(-1, 1)
    rand = rand.to(torch.long) + rowptr[subset].view(-1, 1)
    rand = rand.clamp(max=col.numel() - 1)  # If last node is isolated.

    col = col[rand] if col.numel() > 0 else rand
    col[mask | (count == 0)] = dummy_idx
    return col

def weighted_sample(
    rowptr: Tensor,
    col: Tensor,
    prefix_sum: Tensor,  # prefix sums of the edge weights
    subset: Tensor,
    num_neighbors: int,
    dummy_idx: int,
) -> Tensor:

    device = subset.device
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)

    out_neighbors = []
    for u in subset.tolist():
        start = rowptr[u].item()
        end = rowptr[u+1].item()
        if start == end:
            # Isolated or no neighbors -> dummy
            out_neighbors.append([dummy_idx]*num_neighbors)
            continue

        # Weighted sampling for node u:
        total_weight = prefix_sum[end-1].item() - (prefix_sum[start-1].item() if start > 0 else 0.0)

        # Sample multiple neighbors:
        neighbor_list = []
        for _ in range(num_neighbors):
            r = torch.rand(1).item() * total_weight
            # Convert to actual index via binary search over prefix_sum
            idx = start + binary_search(prefix_sum, r, start, end)
            neighbor_list.append(col[idx].item())

        out_neighbors.append(neighbor_list)

    out_neighbors = torch.tensor(out_neighbors, device=device, dtype=torch.long).view(-1)
    out_neighbors[mask] = dummy_idx  # Mark masked nodes
    return out_neighbors

def binary_search(prefix_sum: Tensor, val: float, left: int, right: int) -> int:
    # Standard binary search on prefix_sum[left:right] to find where val fits.
    # This is a simplified CPU version; for GPU you may want 'torch.searchsorted'.
    # We'll do a minimal CPU-like approach:
    l, r = left, right-1
    base = prefix_sum[left-1].item() if left > 0 else 0.0
    while l < r:
        mid = (l + r) // 2
        if prefix_sum[mid].item() - base < val:
            l = mid + 1
        else:
            r = mid
    return l - left  # relative index from 'start'