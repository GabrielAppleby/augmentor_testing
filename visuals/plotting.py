import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.plot


def label_df(num_instanaces: int, readable_names: np.array) -> pd.DataFrame:
    return pd.DataFrame({'index': np.arange(num_instanaces), 'label': readable_names})


def save_labeled_plot(trained_map: umap.UMAP,
                      readable_labels: pd.DataFrame,
                      path: str) -> None:
    plot = umap.plot.points(trained_map, labels=readable_labels['label'])
    plot.get_figure().savefig(path)
    # umap.plot does not close its plot
    plt.close()


def save_diagnostic_plot(trained_map: umap.UMAP,
                         coloring_values: np.array,
                         path: str):
    embedding = trained_map.embedding_
    cmap = 'viridis'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    point_size = 100.0 / np.sqrt(embedding.shape[0])

    vmin = np.percentile(coloring_values, 5)
    vmax = np.percentile(coloring_values, 95)
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=point_size,
        c=coloring_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Colored by neighborhood Jaccard index")
    ax.text(
        0.99,
        0.01,
        "UMAP: n_neighbors={}, min_dist={}".format(
            trained_map.n_neighbors, trained_map.min_dist
        ),
        transform=ax.transAxes,
        horizontalalignment="right"
    )
    ax.set(xticks=[], yticks=[])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(coloring_values)
    plt.colorbar(mappable, ax=ax)
    plt.savefig(path)
    plt.close()
