import numpy as np
import umap
import umap.plot
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import jaccard_score

from augmentations.augmentting import scale_augmented_features
from data.data_processor import load_data
from neighborhoods.neighbors import nearest_neighbors
from results.pathing import make_results_path
from visuals.plotting import label_df, save_diagnostic_plot


def plot_by_augmentation_weight_with_constant_neighbors(n_neighbors=6):
    features, labels, readable_names = load_data()

    features_scaled = scale_augmented_features(np.copy(features), augmentation_weight=0.0)
    trained_map = umap.UMAP(metric='cosine', n_neighbors=n_neighbors, random_state=42, init='random', verbose=True).fit(
        features_scaled)
    initial_embedding = trained_map.embedding_
    nn_indices_high_original = nearest_neighbors(n_neighbors, features)
    nn_indices_low_original = nearest_neighbors(n_neighbors, initial_embedding)
    accuracy = umap.plot._nhood_compare(nn_indices_high_original, nn_indices_low_original)

    print(accuracy)
    print(np.average(accuracy))


    next_trained_map = umap.UMAP(metric='cosine', n_neighbors=n_neighbors, random_state=42, init=initial_embedding, verbose=True).fit(
        features_scaled)
    next_initial_embedding = next_trained_map.embedding_
    nn_indices_next = nearest_neighbors(n_neighbors, next_initial_embedding)

    accuracy = umap.plot._nhood_compare(nn_indices_high_original, nn_indices_next)
    print(accuracy)
    print(np.average(accuracy))
    # print(np.min(accuracy))
    # save_diagnostic_plot(trained_map, accuracy,
    #               make_results_path('jaccard_comparing_to_original_low_d', n_neighbors, 0.0))

    # for augmentation_weight in np.arange(0.5, 20.0, .5):
    #     features_scaled = scale_augmented_features(np.copy(features), augmentation_weight)
    #     trained_map: umap.UMAP = umap.UMAP(init=initial_embedding, metric='cosine',
    #                                        n_neighbors=n_neighbors, random_state=42).fit(
    #         features_scaled)
    #     initial_embedding = trained_map.embedding_
    #     nn_indices_blah = nearest_neighbors(n_neighbors, initial_embedding)
    #     accuracy = umap.plot._nhood_compare(nn_indices_no_augmentation, nn_indices_blah)
    #     save_diagnostic_plot(trained_map, accuracy,
    #                          make_results_path('jaccard_comparing_to_original_high_d', n_neighbors,
    #                                            augmentation_weight))


def main():
    plot_by_augmentation_weight_with_constant_neighbors()


if __name__ == '__main__':
    main()
