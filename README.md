# Building more effective user embeddings with Metapath2vec

This repository explores how we can build user embeddings in a more effective way than is currently done in production at SoundCloud. Currently, Word2Vec is trained on listening histories for each user which gives us an embedding per SoundCloud track. These embeddings are used downstream for various tasks. For example Autoplay (Non-personalized recommendation) takes a given track a user just interacted with and recommends the top k most similar songs in embeddings space (KNN). Curated playlists are created by using the average of a users listened to tracks as a user embeddings, and then we find the k most similar tracks to the user's embedding. This is a very computationally easy solution, but my hypothesis is that using a simple weighted average loses a lot of rich information that could be stored in the user embedding.

In this notebook we introduce [Metapath2Vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) as an alternative to creating the user embeddings. Metapath2Vec is a scalable heterogeneous graph method which uses metapath based random walk sampling to create neighborhood sequences for each node in the graph, which are then fed to W2V, resulting in embeddings for each node in the same space which are low-dimensional representations of their neighborhood. It is expected that the new track/video embeddings would not perform any better than the original Word2Vec embeddings, but for personalized recommendation using user embeddings, it could work better than the average because we are explicitly learning these embeddings.

We also attempt freezing the track/video embeddings to those learned by W2V, and only allowing the user embeddings to adapt, but the performance went to 0 indicating that metapath2vec requires all embeddings to be changed in order to output high quality predictions.


## Dataset
As a proof of concept we use the open source dataset [KuaiRec](https://kuairec.com/), which provides us with data similar to SoundClouds. We are provided video streaming interaction data for approx. 7000 users across approx. 10000 videos, as well as a social follows graph between users. We are also given a fully observed test set which allows us to be confident in the performance of any systems trained (removes the conterfactual problem in recommendations)

To run this notebook, please download the KuaiRec dataset from the above site, unzip it into this directory, so you have a repository structure similar to:

Metapath2Vec-POC/  
├── KuaiRec/  
│   ├── big_matrix.csv  
│   ├── small_matrix.csv  
│   ├── social_network.csv  
├── Metapath2Vec.ipynb  
├── README.md  