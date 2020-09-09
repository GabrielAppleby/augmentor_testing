import numpy as np
import umap
import umap.plot
from sklearn.neighbors import NearestNeighbors

from augmentations.augmentting import scale_augmented_features
from data.data_processor import load_data
from neighborhoods.neighbors import nearest_neighbors
from results.pathing import make_results_path
from visuals.plotting import label_df, save_labeled_plot


def plot_by_augmentation_weight_with_constant_neighbors_and_intialization(n_neighbors=6):
    features, labels, readable_names = load_data()

    readable_labels = label_df(features.shape[0], readable_names)

    features_scaled = scale_augmented_features(np.copy(features), augmentation_weight=0.0)
    nn_indices_no_augmentation = nearest_neighbors(n_neighbors, features_scaled)
    trained_map = umap.UMAP(metric='cosine', n_neighbors=n_neighbors, random_state=42).fit(
        features_scaled)
    save_labeled_plot(trained_map, readable_labels,
                      make_results_path('constant_neighborhood', n_neighbors, augmentation_weight=0.0))

    for augmentation_weight in np.arange(0.0001, .002, 0.0001):
        features_scaled = scale_augmented_features(np.copy(features), augmentation_weight)
        nn_indices_with_augmentation = nearest_neighbors(features_scaled)
        if not np.array_equal(nn_indices_no_augmentation, nn_indices_with_augmentation):
            break

        trained_map = umap.UMAP(metric='cosine', n_neighbors=n_neighbors, random_state=42).fit(
            features_scaled)
        save_labeled_plot(trained_map, readable_labels,
                          make_results_path('constant_neighborhood', n_neighbors, augmentation_weight))


def main():
    plot_by_augmentation_weight_with_constant_neighbors_and_intialization()


if __name__ == '__main__':
    main()
