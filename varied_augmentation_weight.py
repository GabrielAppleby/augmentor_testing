import numpy as np
import umap
import umap.plot

from augmentations.augmentting import scale_augmented_features
from data.data_processor import load_data
from results.pathing import make_results_path
from visuals.plotting import label_df, save_labeled_plot


def plot_by_augmentation_weight(n_neighbors=6) -> None:
    features, labels, readable_names = load_data()

    readable_labels = label_df(features.shape[0], readable_names)

    for augmentation_weight in np.arange(0.0, 20.0, .5):
        features_scaled = scale_augmented_features(np.copy(features), augmentation_weight)
        trained_map: umap.UMAP = umap.UMAP(metric='cosine', n_neighbors=n_neighbors, random_state=42).fit(features_scaled)
        save_labeled_plot(trained_map, readable_labels,
                          make_results_path('varied_augmentation_weight', n_neighbors,
                                    augmentation_weight))


def main():
    plot_by_augmentation_weight()


if __name__ == '__main__':
    main()
