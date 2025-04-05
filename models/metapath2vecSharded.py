# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.types import DataType
from typing import Dict, List, Tuple, Optional


EPS = 1e-15

# Random walk sampling
def sample(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    rowcount: torch.Tensor,
    subset: torch.Tensor,
    num_neighbors: int,
    dummy_idx: int,
) -> torch.Tensor:
    """
    Simple random-walk single neighbor sampling.
    """
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = torch.rand((subset.size(0), num_neighbors), device=subset.device)
    rand *= count.to(rand.dtype).view(-1, 1)
    rand = rand.long() + rowptr[subset].view(-1, 1)
    rand = rand.clamp(max=col.numel() - 1)

    col_out = col[rand] if col.numel() > 0 else rand
    col_out[mask | (count == 0)] = dummy_idx
    return col_out

class ShardedMetaPath2Vec(nn.Module):
    def __init__(
        self,
        adj_data: Dict[str, Dict[Tuple[str,str,str], torch.Tensor]],
        embedding_dim: int,
        metapath: List[Tuple[str,str,str]],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
    ):
        super().__init__()

        self.rowptr_dict = adj_data["rowptr_dict"]
        self.col_dict = adj_data["col_dict"]
        self.rowcount_dict = adj_data["rowcount_dict"]
        self.num_nodes_dict = adj_data["num_nodes_dict"]

        # Validate
        for e1, e2 in zip(metapath[:-1], metapath[1:]):
            if e1[-1] != e2[0]:
                raise ValueError("Invalid metapath chain.")
        if walk_length + 1 < context_size:
            raise ValueError("walk_length + 1 must be >= context_size")

        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        # Round actual embedding dim to closest multiple of 4 (required by torchrec)
        if embedding_dim % 4 == 0:
            self.effective_dim = embedding_dim
            self.actual_dim = embedding_dim
        else:
            self.effective_dim = embedding_dim
            self.actual_dim = ((embedding_dim + 3) // 4) * 4

        # ID space
        types_set = set([m[0] for m in metapath] + [m[-1] for m in metapath])
        types = sorted(list(types_set))
        count = 0
        self.start, self.end = {}, {}
        for t in types:
            self.start[t] = count
            count += self.num_nodes_dict[t]
            self.end[t] = count

        self.dummy_idx = count
        total_count = count + 1
        self.total_count = total_count

        # offset array for random walk
        offset = [self.start[metapath[0][0]]]
        rep_times = (walk_length // len(metapath)) + 1
        offset += [self.start[e[-1]] for e in metapath] * rep_times
        offset = offset[:walk_length + 1]
        self.offset = torch.tensor(offset, dtype=torch.long)

        # Sharded embedding table
        self.ec = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name="global_table",
                    embedding_dim=self.actual_dim,
                    num_embeddings=total_count,
                    data_type=DataType.FP16
                )
            ],
            device=torch.device("meta"),  # placed on cuda later
        )

    def loader(self, **loader_kwargs):
        dataset = range(self.num_nodes_dict[self.metapath[0][0]])
        return DataLoader(dataset, collate_fn=self._sample, **loader_kwargs)
    
    def _sample(self, batch_list):
        if not isinstance(batch_list, torch.Tensor):
            batch = torch.tensor(batch_list, dtype=torch.long)
        else:
            batch = batch_list

        pos_rw = self._pos_sample(batch)
        neg_rw = self._neg_sample(batch)
        return (self._walks_to_kjt(pos_rw), self._walks_to_kjt(neg_rw))
    
    def _pos_sample(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.repeat(self.walks_per_node)
        rws = [batch]
        for i in range(self.walk_length):
            e_type = self.metapath[i % len(self.metapath)]
            nxt = sample(
                self.rowptr_dict[e_type],
                self.col_dict[e_type],
                self.rowcount_dict[e_type],
                rws[-1],
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
            ).view(-1)
            rws.append(nxt)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw.clamp_(max=self.dummy_idx)

        slides = 1 + self.walk_length + 1 - self.context_size
        outs = []
        for j in range(slides):
            outs.append(rw[:, j : j + self.context_size])
        return torch.cat(outs, dim=0)
    
    def _neg_sample(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        rws = [batch]
        for i in range(self.walk_length):
            e_type = self.metapath[i % len(self.metapath)]
            # random negative
            nxt = torch.randint(
                0,
                self.num_nodes_dict[e_type[-1]],
                (batch.size(0),),
                dtype=torch.long,
                device=batch.device,
            )
            rws.append(nxt)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw.clamp_(max=self.dummy_idx)

        slides = 1 + self.walk_length + 1 - self.context_size
        outs = []
        for j in range(slides):
            outs.append(rw[:, j : j + self.context_size])
        return torch.cat(outs, dim=0)
    
    # Convert walks to KeyedJaggedTensor (Sharded embedding lookup tensor)
    def _walks_to_kjt(self, walks: torch.Tensor) -> KeyedJaggedTensor:
        B, C = walks.shape
        vals = walks.view(-1)
        lengths = torch.full((B,), C, dtype=torch.int32)
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["global_table"],
            values=vals,
            lengths=lengths,
        )
        return kjt
    
    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ec_out = self.ec(kjt)
        out = ec_out["global_table"].values()
        out = out[:, : self.effective_dim]     # ignore dummy dims
        return out
    
    # Original metapath2vec skip-gram loss
    def skip_gram_loss(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        vals = self.forward(kjt) # Gets embeddings in sample
        B = kjt.lengths().numel()
        C = kjt.lengths()[0].item()
        vals = vals.view(B, C, self.effective_dim)

        start_emb = vals[:, 0:1, :]
        rest_emb  = vals[:, 1:, :]
        out = (start_emb * rest_emb).sum(dim=-1).view(-1)
        return out

    # Original metapath2vec skip-gram loss with negative sampling
    def loss(self, pos_kjt: KeyedJaggedTensor, neg_kjt: KeyedJaggedTensor) -> torch.Tensor:
        out_pos = self.skip_gram_loss(pos_kjt)
        pos_loss = -torch.log(torch.sigmoid(out_pos) + 1e-15).mean()

        out_neg = self.skip_gram_loss(neg_kjt)
        neg_loss = -torch.log(1 - torch.sigmoid(out_neg) + 1e-15).mean()

        return pos_loss + neg_loss
    
    # Initialize pretrained embeddings (W2V, etc.)
    def init_pretrained(
        self, 
        node_type: str, 
        pretrained_weight: torch.Tensor,
    ):
        """
        Copy a pretrained embedding (size [num_nodes_of_type, effective_dim])
        into the appropriate slice in the big global_table. 
        The leftover dims remain unused or zero.
        """

        start_idx = self.start[node_type]
        end_idx   = self.end[node_type]
        row_count = end_idx - start_idx
        if pretrained_weight.size(0) != row_count or pretrained_weight.size(1) != self.effective_dim:
            raise ValueError("Shape mismatch for pretrained")

        param = next(self.ec.parameters())
        with torch.no_grad():
            param[start_idx:end_idx, : self.effective_dim].copy_(pretrained_weight)

    # Gather sharded embeddings to CPU for evaluation
    def get_embeddings(self, item: str = "video") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        rank = dist.get_rank()
        
        # Use the default process group, switch to Gloo for CPU ops if needed
        default_pg = dist.group.WORLD
        if dist.get_backend(default_pg) != "gloo":
            gloo_pg = dist.new_group(backend="gloo")
        else:
            gloo_pg = default_pg

        # Get the sharded state dict
        sd = self.state_dict()
        global_sharded = sd["ec.embeddings.global_table.weight"]  # Sharded embedding table
        
        out_size = (self.total_count, self.actual_dim)
        # Allocate tensor on CPU, None for other devices
        out_tensor = (
            torch.empty(out_size, dtype=torch.float16, device="cpu")
            if rank == 0
            else None
        )
        # Sync all devices before gathering
        dist.barrier(group=gloo_pg)

        # Move sharded tensor to CPU and gather to rank 0 (all devices participate)
        global_sharded_cpu = global_sharded.cpu(process_group=gloo_pg)
        global_sharded_cpu.gather(out=out_tensor)  # Collect operation

        # Sync again to ensure gather is complete
        dist.barrier(group=gloo_pg)
        if gloo_pg != default_pg:
            dist.destroy_process_group(gloo_pg)

        # Only rank 0 (cpu) processes the gathered tensor
        if rank == 0:
            # Slice out user & item embeddings, ignore dummy dimensions
            user_start, user_end = self.start['user'], self.end['user']
            item_start, item_end = self.start[item], self.end[item]
            user_cpu = out_tensor[user_start:user_end, :self.effective_dim]
            item_cpu = out_tensor[item_start:item_end, :self.effective_dim]
            return (user_cpu, item_cpu)
        else:
            return (None, None)
