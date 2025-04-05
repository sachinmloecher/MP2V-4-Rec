import os
import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import (
    ShardingEnv,
    ShardingType,
    ShardingPlan,
    ModuleSharder,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
)
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed.optim import _apply_optimizer_in_backward as apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.optim.keyed import KeyedOptimizerWrapper
from multiprocessing import Manager


from models.baseRecommender import Recommender
from models.metapath2vecSharded import ShardedMetaPath2Vec
from utils.evaluate import kuairec_eval, soundcloud_eval
import psutil
import pickle

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
            
def log_memory(prefix):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"{prefix}: {mem:.2f} GB RAM")

def load_and_prepare_adj_data(path="SoundCloud/adj_data_90_sampled.pkl"):
    """
    1) Loads adjacency from disk (pickled).
    2) Calls .share_memory_() on each Tensor so they reside in OS shared memory.
    Returns the resulting adjacency dictionary.
    """
    with open(path, "rb") as f:
        adj_data = pickle.load(f)

    # Convert Tensors to shared memory
    for key, t in adj_data['rowptr_dict'].items():
        t.share_memory_()
    for key, t in adj_data['col_dict'].items():
        t.share_memory_()
    for key, t in adj_data['rowcount_dict'].items():
        t.share_memory_()

    # You might have edge_weight_dict too
    if 'edge_weight_dict' in adj_data:
        for key, t in adj_data['edge_weight_dict'].items():
            if t is not None:
                t.share_memory_()

    return adj_data

class ShardedMetaPath2VecRecommender(Recommender):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    def fit(self, w2v_embeddings, metapath, val_data, test_data):
        self.evaluate = soundcloud_eval if self.dataset == 'SoundCloud' else kuairec_eval
        # Row-wise embedding sharding
        constraints = {
        "global_table": ParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
        )
        }
        adj_data = load_and_prepare_adj_data()
        world_size = torch.cuda.device_count()
        with Manager() as manager:
            results = manager.dict()
            shared_val_data = manager.dict(val_data)
            shared_test_data = manager.dict(test_data)
            mp.spawn(_worker,
                    nprocs=world_size,
                    join=True,
                    args=(
                        world_size,
                        adj_data,
                        metapath,
                        self.config,
                        constraints,
                        shared_val_data,
                        shared_test_data,
                        w2v_embeddings,
                        self.evaluate,
                        "nccl",
                        results
                    )
            )
            if 0 in results:
                result = results[0]
                return {'Epoch': result['Epoch'], 'val_metrics': result['val_metrics']}
            else:
                return {'Epoch': 0, 'val_metrics': None}


def _worker(rank: int, 
            world_size: int, 
            adj_data, 
            metapath,
            config,
            constraints, 
            val_data, 
            test_data,
            w2v_embeddings=None,
            evaluate=None,
            backend="nccl",
            results=None):
    item = 'track' if config['dataset'] == 'SoundCloud' else 'video'
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = ShardedMetaPath2Vec(
        adj_data=adj_data,
        embedding_dim=config['embedding_dim'],
        metapath=metapath,
        walk_length=config['walk_length'],
        context_size=config['context_size'],
        walks_per_node=config['walks_per_node'],
        num_negative_samples=config['num_negative_samples'],
    )
    def print_embedding_sizes(model):
        total_params = 0
        for name, param in model.named_parameters():
            # param.numel() = number of scalars
            # param.element_size() = bytes per element
            size_mb = (param.numel() * param.element_size()) / 1024**2
            print(f"{name} -> {size_mb:.2f} MB, dtype={param.dtype}, shape={tuple(param.shape)}")
            total_params += param.numel() * param.element_size()
        print(f"Total param size: {total_params / 1024**2:.2f} MB")
    print_embedding_sizes(model)
    if w2v_embeddings:
        model.init_pretrained(item, w2v_embeddings[0])
        model.init_pretrained('user', w2v_embeddings[1])
    # Make gradient updates be applied in backward pass
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        model.ec.parameters(),
        {"lr": config['lr']},
    )
    # Sharding plan
    topology = Topology(world_size=world_size, compute_device="cuda")
    sharders = [EmbeddingCollectionSharder()]
    planner = EmbeddingShardingPlanner(topology=topology, constraints=constraints)
    plan: ShardingPlan = planner.collective_plan(model, sharders, dist.group.WORLD)
    # Distribute model
    dmp_model = DMP(
        module=model,
        env=ShardingEnv.from_process_group(dist.group.WORLD),
        plan=plan,
        sharders=sharders,
        device=device,
    )
    optimizer = KeyedOptimizerWrapper(
        dict(dmp_model.named_parameters()),
        lambda params: torch.optim.Adam(params, lr=config['lr']),
    )
    loader = dmp_model.module.loader(batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    print(len(loader))
    best_recall_so_far = 0.0
    best_val_metrics = None
    best_epoch = 0
    epochs_no_improve = 0
    should_stop = torch.tensor([0], dtype=torch.int32, device=device)
    log_memory(f"Rank {rank}, thread created, before training loop")
    try:
        for epoch in range(1, config['max_epochs'] + 1):
            dmp_model.train()
            total_loss = 0.0
            for i, (pos_kjt, neg_kjt) in enumerate(loader):
                print(i)
                pos_kjt = pos_kjt.to(device)
                neg_kjt = neg_kjt.to(device)
                optimizer.zero_grad()
                loss_val = dmp_model.module.loss(pos_kjt, neg_kjt)
                loss_val.backward()
                optimizer.step()
                total_loss += loss_val.item()
            dist.barrier()
            print(f"Rank {rank}, epoch {epoch}, loss: {total_loss:.4f}")
            if (epoch % config['eval_freq'] == 0):
                print(f"Rank {rank}, evaluating...")
                dmp_model.eval()
                with torch.no_grad():
                    user_emb, item_emb = dmp_model.module.get_embeddings()
                    if rank == 0: # cpu
                        val_metrics = evaluate(
                            user_emb=user_emb,
                            item_emb=item_emb,
                            val_data=val_data,
                            test_data=test_data,
                            is_validation=True,
                            top_k=config['k'],
                            progress_bar=False,
                        )
                        recall_k = val_metrics.get(f"Recall@{max(config['k'])}", 0.0)
                        print(f"Rank {rank}, epoch {epoch}, recall@{max(config['k'])}: {recall_k:.4f}")
                        print(val_metrics)
                        # Early stopping check (only if we evaluated this epoch)
                        if val_metrics is not None:
                            if recall_k > best_recall_so_far:
                                best_recall_so_far = recall_k
                                best_val_metrics = val_metrics
                                best_epoch = epoch
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1

                            if epochs_no_improve >= config['patience']:
                                should_stop[0] = 1
                dist.broadcast(should_stop, src=0)
                if should_stop.item():
                    break
            dist.barrier()
        if rank == 0:
            results[rank] = {'Epoch': best_epoch, 'val_metrics': best_val_metrics}
        else:
            results[rank] = {'Epoch': 0, 'val_metrics': None}
        dist.barrier()
        dist.destroy_process_group()
        return {'Epoch': best_epoch, 'val_metrics': best_val_metrics}
    except Exception as e:
        raise e        
