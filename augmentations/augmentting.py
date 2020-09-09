import numpy as np


def scale_augmented_features(features: np.array, augmentation_weight: float) -> np.array:
    features[:, -2:] = features[:, -2:] * augmentation_weight
    return features
