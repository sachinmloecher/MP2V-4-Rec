# MP2V-4-Rec: Scalable Graph-Based Recommendation Systems

## Project Summary
This project explores scalable graph-based representation learning for personalized recommendations on large-scale streaming platforms, focusing on SoundCloud and Kuaishou (KuaiRec dataset). Traditional methods like Word2Vec often fail to capture complex user-content and social interactions. We introduce **Metapath2Vec (MP2V)** to recommendation research, extending it with weighted random walks (R-MP2V) to account for interaction quality. Leveraging SoundCloud’s production-scale data—9 million users, 13 million tracks, and over 21 million social connections—and the reproducible KuaiRec dataset, we compare MP2V and R-MP2V against baselines and state-of-the-art methods like UltraGCN. Ablation studies on KuaiRec evaluate social relationships, embedding initialization, and metapath sampling. MP2V delivers competitive performance, is highly scalable, and is being deployed at SoundCloud for playlist curation and personalization. This marks the first application of MP2V in a production-scale music recommender system, offering practical insights for real-world streaming applications.

## Key Figures
Below are selected visualizations from the results:

- **SoundCloud PCA Embeddings**: Comparing W2V and MP2V embedding spaces on SoundCloud (9M users, 13M tracks).
  ![SoundCloud PCA Embeddings](results/SoundCloud/pca_embeddings.png)

- **KuaiRec PCA Embeddings**: Visualizing W2V and MP2V embeddings on KuaiRec.
  ![KuaiRec PCA Embeddings](results/KuaiRec/pca_embeddings.png)

- **SoundCloud Diversity Histogram**: Diversity distribution (1 - avg cosine similarity) of top-100 recommendations for 200,000 users.
  ![SoundCloud Diversity Histogram](results/SoundCloud/diversity_histogram.png)

- **SoundCloud Feature Diversity**: Unique artists and genres in top-100 recommendations.
  ![SoundCloud Feature Diversity](results/SoundCloud/feature_diversity.png)

- **SoundCloud Recall@K and NDCG@K Trends**: Performance across K values (10, 20, 50, 100).
  ![SoundCloud Metrics by K](results/SoundCloud/soundcloud_metrics_by_k.png)

## Setup
To replicate the environment used in this project, use the provided `environment.yaml` file:
```bash
conda env create -f environment.yaml
conda activate mp2v4rec