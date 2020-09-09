import os
from typing import List, Tuple

import numpy as np
# import spacy
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC
# from sklearn.utils import Bunch

DATA_FILE_NAME: str = 'newsgroup.npz'
DATA_FOLDER_PATH: str = os.path.dirname(os.path.abspath(__file__))

SCI_KEY: str = 'sci'
TALK_KEY: str = 'talk'


def load_data() -> Tuple[np.array, np.array, np.array]:
    data = np.load(os.path.join(DATA_FOLDER_PATH, DATA_FILE_NAME))
    features_test = data['features_test']
    feature_augment_one = data['feature_augment_one']
    feature_augment_two = data['feature_augment_two']
    targets_test = data['targets_test']
    readable_targets_test = data['readable_targets_test']

    features_test = np.hstack((features_test, feature_augment_one, feature_augment_two))

    return features_test, targets_test, readable_targets_test

#
# def load_raw_data() -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
#     newsgroups_train: Bunch = fetch_20newsgroups(subset='train',
#                                                  remove=('headers', 'footers', 'quotes'))
#
#     newsgroups_test: Bunch = fetch_20newsgroups(subset='test',
#                                                 remove=('headers', 'footers', 'quotes'))
#
#     text_train = newsgroups_train.data
#     text_test = newsgroups_test.data
#
#     targets_train = newsgroups_train.target
#     targets_test = newsgroups_test.target
#
#     readable_targets_train = [newsgroups_train.target_names[x] for x in targets_train]
#     readable_targets_test = [newsgroups_test.target_names[x] for x in targets_test]
#
#     return text_train, targets_train, readable_targets_train, text_test, targets_test, readable_targets_test
#
#
# def process_text_spacy(text_train: np.array, text_test: np.array) -> Tuple[np.array, np.array]:
#     nlp: spacy.language.Language = spacy.load("en_core_web_lg")
#     nlp.disable_pipes('parser', 'ner')
#
#     vecs_train: List[np.array] = [x.vector for x in list(nlp.pipe(text_train))]
#     mat_train = np.vstack(vecs_train)
#
#     vecs_test: List[np.array] = [x.vector for x in list(nlp.pipe(text_test))]
#     mat_test = np.vstack(vecs_test)
#
#     return mat_train, mat_test
#
#
# def normalize_features(features_train: np.array, features_test: np.array) -> np.array:
#     scaler = MinMaxScaler()
#     features_train = scaler.fit_transform(features_train)
#     features_test = scaler.transform(features_test)
#
#     return features_train, features_test
#
#
# def save_processed_data(features_test: np.array,
#                         feature_augment_one: np.array,
#                         feature_augment_two: np.array,
#                         targets_test: np.array,
#                         readable_targets_test: np.array) -> None:
#     np.savez(DATA_FILE_NAME,
#              features_test=features_test,
#              feature_augment_one=feature_augment_one,
#              feature_augment_two=feature_augment_two,
#              targets_test=targets_test,
#              readable_targets_test=readable_targets_test)
#
#
# def get_sci_topic_indicator_arr(targets: np.array) -> np.array:
#     return (targets == 11) | (targets == 12) | (targets == 13) | (targets == 14)
#
#
# def get_talk_topic_indicator_arr(targets: np.array) -> np.array:
#     return (targets == 16) | (targets == 17) | (targets == 18) | (targets == 19)
#
#
# def predict_topic_membership_proba(features_train: np.array, labels_train: np.array,
#                                    features_test: np.array) -> np.array:
#     svc = SVC(probability=True)
#
#     svc.fit(X=features_train, y=labels_train)
#
#     return svc.predict_proba(features_test)[:, -1].reshape((-1, 1))
#
#
# def main():
#     text_train, targets_train, readable_targets_train, text_test, targets_test, readable_targets_test = load_raw_data()
#
#     features_train, features_test = process_text_spacy(text_train, text_test)
#     features_train, features_test = normalize_features(features_train, features_test)
#
#     sci_indicator_train = get_sci_topic_indicator_arr(targets_train)
#     talk_indicator_train = get_talk_topic_indicator_arr(targets_train)
#
#     membership_probas = {}
#     for class_key, indicator_train in [(SCI_KEY, sci_indicator_train),
#                                        (TALK_KEY, talk_indicator_train)]:
#         membership_probas[class_key] = predict_topic_membership_proba(features_train,
#                                                                       indicator_train,
#                                                                       features_test)
#
#     save_processed_data(features_test,
#                         membership_probas[SCI_KEY],
#                         membership_probas[TALK_KEY],
#                         targets_test,
#                         readable_targets_test)
#
#
# if __name__ == '__main__':
#     main()
