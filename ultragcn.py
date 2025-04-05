import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time

# Step 1: Data Preparation
with open("KuaiRec/adj_data.pkl", "rb") as f:
    adj_data = pickle.load(f)
with open("KuaiRec/val_data.pkl", "rb") as f:
    val_data_dict = pickle.load(f)
with open("KuaiRec/test_data.pkl", "rb") as f:
    test_data_dict = pickle.load(f)

num_users = adj_data['num_nodes_dict']['user']
num_items = adj_data['num_nodes_dict']['video']

rowptr = adj_data['rowptr_dict'][('user', 'watches', 'video')]
col = adj_data['col_dict'][('user', 'watches', 'video')]
train_data = []
for u in range(num_users):
    start = rowptr[u]
    end = rowptr[u + 1]
    items = col[start:end].tolist()
    train_data.extend([(u, i) for i in items])

train_mat = sp.coo_matrix(
    (np.ones(len(train_data)), ([x[0] for x in train_data], [x[1] for x in train_data])),
    shape=(num_users, num_items)
).tocsr()

items_D = np.array(train_mat.sum(axis=0)).flatten()
users_D = np.array(train_mat.sum(axis=1)).flatten()
beta_uD = (np.sqrt(users_D + 1) / (users_D + 1e-8)).reshape(-1)
beta_iD = (1 / np.sqrt(items_D + 1)).reshape(-1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
constraint_mat = {
    "beta_uD": torch.from_numpy(beta_uD).float().to(device),
    "beta_iD": torch.from_numpy(beta_iD).float().to(device)
}

def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero=False):
    print('Computing item-item constraint matrix efficiently...')
    A = train_mat.T.dot(train_mat).tocsr()
    if ii_diagonal_zero:
        A.setdiag(0)
    A.eliminate_zeros()
    n_items = A.shape[0]
    items_D = np.array(A.sum(axis=0)).flatten()
    users_D = np.array(A.sum(axis=1)).flatten()
    beta_uD = (np.sqrt(users_D + 1) / (users_D + 1e-8)).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = beta_uD.dot(beta_iD)
    res_mat = torch.zeros((n_items, num_neighbors), dtype=torch.long)
    res_sim_mat = torch.zeros((n_items, num_neighbors), dtype=torch.float32)
    batch_size = 10000
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch_A = A[start:end].toarray()
        batch_weighted = batch_A * all_ii_constraint_mat[start:end]
        batch_tensor = torch.from_numpy(batch_weighted).float()
        row_sims, row_idxs = torch.topk(batch_tensor, k=num_neighbors, dim=1)
        res_mat[start:end] = row_idxs
        res_sim_mat[start:end] = row_sims
        print(f'Processed items {start} to {end-1}')
    print('Item-item constraint matrix computed!')
    return res_mat.to(device), res_sim_mat.to(device)

ii_neighbor_num = 10
ii_cons_mat_path = 'KuaiRec_ii_constraint_mat.pkl'
ii_neigh_mat_path = 'KuaiRec_ii_neighbor_mat.pkl'
if os.path.exists(ii_cons_mat_path) and os.path.exists(ii_neigh_mat_path):
    with open(ii_cons_mat_path, 'rb') as f:
        ii_constraint_mat = pickle.load(f).to(device)
    with open(ii_neigh_mat_path, 'rb') as f:
        ii_neighbor_mat = pickle.load(f).to(device)
else:
    ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)
    with open(ii_cons_mat_path, 'wb') as f:
        pickle.dump(ii_constraint_mat, f)
    with open(ii_neigh_mat_path, 'wb') as f:
        pickle.dump(ii_neighbor_mat, f)

interacted_items = [[] for _ in range(num_users)]
for u, i in train_data:
    interacted_items[u].append(i)

mask = torch.zeros(num_users, num_items).to(device)
for u, items in enumerate(interacted_items):
    mask[u, items] = -np.inf

train_data_tensor = torch.tensor(train_data, dtype=torch.long)
train_loader = DataLoader(
    train_data_tensor, batch_size=8192, shuffle=True, num_workers=16, pin_memory=True
)
val_loader = DataLoader(
    torch.tensor(list(val_data_dict.keys()), dtype=torch.long), batch_size=1024, shuffle=False, num_workers=5, pin_memory=True
)
test_loader = DataLoader(
    torch.tensor(list(test_data_dict.keys()), dtype=torch.long), batch_size=1024, shuffle=False, num_workers=5, pin_memory=True
)

# Step 2: Model Definition
class UltraGCN(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']
        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = params['initial_weight']
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items])
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items), device=device)
        if self.w4 > 0:
            neg_weight = torch.mul(
                torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                self.constraint_mat['beta_iD'][neg_items.flatten()]
            )
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1), device=device)
        return torch.cat((pos_weight, neg_weight))

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)
        neg_labels = torch.zeros(neg_scores.size(), device=device)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, neg_labels, weight=omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none'
        ).mean(dim=-1)
        pos_labels = torch.ones(pos_scores.size(), device=device)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)], reduction='none'
        )
        return (pos_loss + neg_loss * self.negative_weight).sum()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items])
        sim_scores = self.ii_constraint_mat[pos_items]
        user_embeds = self.user_embeds(users).unsqueeze(1)
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def test_forward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device

# Step 3: Training Loop
params = {
    'user_num': num_users,
    'item_num': num_items,
    'embedding_dim': 50,
    'w1': 1.0,
    'w2': 1.0,
    'w3': 1.0,
    'w4': 1.0,
    'negative_weight': 1.0,
    'gamma': 1e-5,
    'lambda': 1e-5,
    'initial_weight': 0.1,
    'lr': 0.01,
    'batch_size': 8192,
    'max_epoch': 100,
    'early_stop_epoch': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'topk': [100],
    'negative_num': 5,
    'sampling_sift_pos': True,
    'is_validation': True
}

def Sampling(users, pos_items, item_num, neg_ratio, interacted_items, sampling_sift_pos):
    neg_candidates = np.arange(item_num)
    if sampling_sift_pos:
        neg_items = []
        for u in users:
            probs = np.ones(item_num)
            probs[interacted_items[u]] = 0
            probs /= np.sum(probs)
            u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True)
            neg_items.append(u_neg_items)
        neg_items = np.array(neg_items)
    else:
        neg_items = np.random.choice(neg_candidates, (len(users), neg_ratio), replace=True)
    return users, pos_items, torch.from_numpy(neg_items).long().to(users.device)

def evaluate(model, loader, ground_truth_dict, mask, other_dict, top_k, is_validation=True):
    total_recall = {k: 0.0 for k in top_k}
    total_ndcg = {k: 0.0 for k in top_k}
    recommended_items = {k: set() for k in top_k}
    total_items = model.item_num
    user_count = 0

    idcg_cache = {}
    def get_idcg(r_size, k):
        cut = min(k, r_size)
        if (cut, k) not in idcg_cache:
            val = 0.0
            for i in range(cut):
                val += 1.0 / np.log2(i + 2)
            idcg_cache[(cut, k)] = val
        return idcg_cache[(cut, k)]

    with torch.no_grad():
        model.eval()
        for batch_users in loader:
            batch_users = batch_users.to(model.get_device())
            rating = model.test_forward(batch_users)  # Shape: [batch_size, item_num]
            rating += mask[batch_users]  # Mask training items
            rating = rating.cpu().numpy()

            for i, u in enumerate(batch_users.cpu().numpy()):
                relevant_set = set(ground_truth_dict.get(u, []))
                if not relevant_set:
                    continue
                other_set = set(other_dict.get(u, []))
                scores = rating[i]
                # Exclude training and other_set items
                exclude_items = set(interacted_items[u]) | other_set
                valid_items = [vid for vid in range(total_items) if vid not in exclude_items]
                scores_filtered = scores[valid_items]
                ranked_items = np.argsort(-scores_filtered)[:max(top_k)]  # Top-k after exclusion
                ranked_items = [valid_items[vid] for vid in ranked_items]

                hits_positions = [pos for pos, vid in enumerate(ranked_items) if vid in relevant_set]

                for k in top_k:
                    hits_count = sum(1 for pos in hits_positions if pos < k)
                    recall_k = hits_count / float(len(relevant_set))
                    total_recall[k] += recall_k

                    dcg_val = sum(1.0 / np.log2(pos + 2) for pos in hits_positions if pos < k)
                    idcg_val = get_idcg(len(relevant_set), k)
                    ndcg_k = (dcg_val / idcg_val) if idcg_val > 0 else 0.0
                    total_ndcg[k] += ndcg_k

                    recommended_items[k].update(ranked_items[:k])
                user_count += 1

    if user_count == 0:
        return {f'Recall@{k}': 0.0 for k in top_k} | {f'NDCG@{k}': 0.0 for k in top_k} | {f'Coverage@{k}': 0.0 for k in top_k}

    avg_recall = {f'Recall@{k}': total_recall[k] / user_count for k in top_k}
    avg_ndcg = {f'NDCG@{k}': total_ndcg[k] / user_count for k in top_k}
    coverage = {f'Coverage@{k}': len(recommended_items[k]) / total_items for k in top_k}
    return avg_recall | avg_ndcg | coverage

def train(model, optimizer, train_loader, val_loader, test_loader, mask, val_data_dict, test_data_dict, interacted_items, params):
    device = params['device']
    best_recall, best_epoch = 0, 0
    early_stop_count = 0
    patience = 3  # Stop if no improvement for 3 epochs

    for epoch in range(params['max_epoch']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            users, pos_items = batch[:, 0], batch[:, 1]
            users, pos_items, neg_items = Sampling(
                users, pos_items, params['item_num'], params['negative_num'], interacted_items, params['sampling_sift_pos']
            )

            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, val_data_dict, mask, test_data_dict if params['is_validation'] else val_data_dict, params['topk'], is_validation=True)
        recall_at_k = val_metrics[f'Recall@{max(params["topk"])}']
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Validation Metrics: {recall_at_k}")

        if recall_at_k > best_recall:
            best_recall, best_epoch = recall_at_k, epoch
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                #print(f'Early stop at epoch {epoch+1}, Best Recall@{params["topk"]}: {best_recall:.4f} at epoch {best_epoch+1}')
                break
    print(f'Finished training, Best Recall@{max(params["topk"])}: {best_recall:.4f} at epoch {best_epoch+1}')
    return best_recall, best_epoch


""" # Step 5: Run Training
model = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
model = model.to(params['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
train(model, optimizer, train_loader, val_loader, test_loader, mask, val_data_dict, test_data_dict, interacted_items, params)
 """
from itertools import product

# Define hyperparameter grid
param_grid = {
    'negative_num': [1, 5, 10],
    'gamma': [1e-5, 1e-4, 1e-3],
    'lambda': [1e-5, 1e-4, 1e-3]
}

# Base params (unchanging)
base_params = {
    'user_num': num_users,
    'item_num': num_items,
    'lr': 0.005,
    'w1': 1.0,
    'w2': 1.0,
    'w3': 1.0,
    'w4': 1.0,
    'negative_weight': 1.0,
    'initial_weight': 0.1,
    'batch_size': 8192,
    'max_epoch': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'topk': [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'sampling_sift_pos': True,
    'is_validation': True, 
    'embedding_dim': 50
}

best_config = None
best_test_metrics = None
best_recall = 0

for neg_num, gamma, lambda_ in product(
    param_grid['negative_num'],
    param_grid['gamma'], param_grid['lambda']
):
    print(f"\nTesting config: neg_num={neg_num}, gamma={gamma}, lambda={lambda_}")
    params = base_params.copy()
    params.update({
        'negative_num': neg_num,
        'gamma': gamma,
        'lambda': lambda_
    })

    model = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat).to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    val_metrics = train(model, optimizer, train_loader, val_loader, test_loader, mask, val_data_dict, test_data_dict, interacted_items, params)

    recall_at_100 = val_metrics[0]
    if recall_at_100 > best_recall:
        best_recall = recall_at_100
        best_config = (neg_num, gamma, lambda_)
        best_test_metrics = val_metrics

print(f"\nBest config: negative_num={best_config[0]}, gamma={best_config[1]}, lambda={best_config[2]}")
print(f"Best test metrics: {best_test_metrics}")