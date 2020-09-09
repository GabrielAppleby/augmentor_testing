import os

RESULTS_FOLDER: str = os.path.dirname(os.path.realpath(__file__))
NEIGHBORS_TEMPLATE: str = 'n_neighbors{}'
FILENAME_TEMPLATE: str = '{}_weight.png'
DATASET: str = 'newsgroup'


def make_results_path(experiment_name: str, n_neighbors: int, augmentation_weight: float) -> str:
    neighbors = NEIGHBORS_TEMPLATE.format(n_neighbors)
    file_name = FILENAME_TEMPLATE.format(augmentation_weight)
    folder_path = os.path.join(RESULTS_FOLDER, experiment_name, DATASET, neighbors)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return os.path.join(folder_path, file_name)
