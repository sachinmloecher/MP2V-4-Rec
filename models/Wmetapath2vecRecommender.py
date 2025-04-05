from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.index import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from models.baseRecommender import Recommender
from models.Wmetapath2vec import WeightedMetaPath2Vec
from utils.evaluate import kuairec_eval, soundcloud_eval



class WMetaPath2VecRecommender(Recommender):
    def __init__(self, config):
        super().__init__(config)

    def fit(self, adj_data, w2v_embeddings, metapath, val_data, test_data, logger):
        self.evaluate = soundcloud_eval if self.dataset == 'SoundCloud' else kuairec_eval
        item = 'track' if self.dataset == 'SoundCloud' else 'video'
        
        self.model = WeightedMetaPath2Vec(
            adj_data,
            embedding_dim=self.embedding_dim,
            metapath=metapath,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            sparse=True
        ).to(self.device)
        
        if w2v_embeddings:
            with torch.no_grad():
                self.model(item).data.copy_(w2v_embeddings[0])
                self.model('user').data.copy_(w2v_embeddings[1])

        loader = self.model.loader(batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        optimizer = torch.optim.SparseAdam([p for p in self.model.parameters()], lr=self.lr)
        
        def train_one_epoch(epoch):
            """ Trains one epoch and returns total loss """
            self.model.train()
            total_loss = 0
            for step, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
                loss = self.model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            return total_loss
        
        best_recall_so_far = 0.0
        best_val_metrics = None
        best_epoch = 0
        epochs_no_improve = 0


        for epoch in range(1, self.max_epochs + 1):
            loss = train_one_epoch(epoch)

            # Check if we should evaluate on this epoch
            if (epoch % self.eval_freq == 0 and epoch != 0):
                self.model.eval()
                user_emb = self.model('user').cpu().detach().numpy()
                item_emb = self.model(item).cpu().detach().numpy()

                # Evaluate on validation set
                val_metrics = self.evaluate(
                    user_emb=user_emb,
                    item_emb=item_emb,
                    val_data=val_data,
                    test_data=test_data,
                    is_validation=True,
                    top_k=self.k,
                    progress_bar=False
                )
                del user_emb
                del item_emb

                recall_k = val_metrics.get(f'Recall@{max(self.k)}', 0.0)
            else:
                val_metrics = None
                recall_k = "---"  # No evaluation, so we log "---"

            logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Recall@{max(self.k)} = {recall_k}")

            # Early stopping check (only if we evaluated this epoch)
            if val_metrics is not None:
                if recall_k > best_recall_so_far:
                    best_recall_so_far = recall_k
                    best_val_metrics = val_metrics
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    logger.info(f"Stopping early after {epoch} epochs (no improvement in {self.patience} epochs).")
                    break

        return {'Epoch': best_epoch, 'val_metrics': best_val_metrics}