import numpy as np
import umap
import umap.plot

from augmentations.augmentting import scale_augmented_features
from autoencoders.autoencoder import Autoencoder
from autoencoders.models import get_newsgroup_autoencoder
from data.data_processor import load_data
from results.pathing import make_results_path
from visuals.plotting import label_df, save_labeled_plot


def retraining_embedding(n_neighbors=6) -> None:
    features, labels, readable_names = load_data()

    readable_labels = label_df(features.shape[0], readable_names)

    initial_embedding = "spectral"
    autoencoder: Autoencoder = get_newsgroup_autoencoder()

    for augmentation_weight in np.arange(0.0, 20.0, .5):
        features_scaled = scale_augmented_features(np.copy(features), augmentation_weight)
        autoencoder.fit(features_scaled)
        features_scaled = autoencoder.encode(features_scaled)
        trained_map: umap.UMAP = umap.UMAP(init=initial_embedding, metric='cosine',
                                           n_neighbors=n_neighbors, random_state=42).fit(
            features_scaled)
        initial_embedding = trained_map.embedding_
        save_labeled_plot(trained_map, readable_labels,
                          make_results_path('retraining_embedding', n_neighbors,
                                    augmentation_weight))


def main():
    retraining_embedding()


if __name__ == '__main__':
    main()
