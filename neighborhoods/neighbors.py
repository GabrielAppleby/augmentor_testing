import numpy as np

from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(n_neighbors, features_scaled: np.array) -> np.array:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(features_scaled)
    indices = nbrs.kneighbors(return_distance=False)
    return indices