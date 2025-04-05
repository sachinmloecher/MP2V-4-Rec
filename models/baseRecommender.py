from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class Recommender(ABC):
    def __init__(self, config):
        super().__init__()
        for key, value in config.items():
            setattr(self, key, value)
        self.model = None

    @abstractmethod
    def fit(self, data):
        pass