import glob
import logging

import numpy as np
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from sdgym.utils import CATEGORICAL, CONTINUOUS, ORDINAL

logging.basicConfig(level=logging.INFO)


BAYESIAN_PARAMETER = {
    'grid': 30,
    'gridr': 30,
    'ring': 10,
}

DATASET_MODELS_MAP = {
    'mnist12': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (LogisticRegression(
            solver='lbfgs', n_jobs=2, multi_class="auto", class_weight='balanced', max_iter=50),
            "Logistic Regression"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)")
    ],
    'mnist28': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (LogisticRegression(
            solver='lbfgs', n_jobs=2, multi_class="auto", class_weight='balanced', max_iter=50),
            "Logistic Regression"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)")
    ],
    'adult': [
        (DecisionTreeClassifier(max_depth=15, class_weight='balanced'),
            "Decision Tree (max_depth=20)"),
        (AdaBoostClassifier(), "Adaboost (estimator=50)"),
        (LogisticRegression(
            solver='lbfgs', n_jobs=2, class_weight='balanced', max_iter=50),
            "Logistic Regression"),
        (MLPClassifier((50, ), max_iter=50), "MLP (50)")
    ],
    'census': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (AdaBoostClassifier(), "Adaboost (estimator=50)"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)"),
    ],
    'credit': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (AdaBoostClassifier(), "Adaboost (estimator=50)"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)"),
    ],
    'intrusion': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)"),
    ],
    'covtype': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)"),
    ],
    'news': [
        (LinearRegression(), "Linear Regression"),
        (MLPRegressor((100, ), max_iter=50), "MLP (100)")
    ]
}


def get_models(dataset):
    models = DATASET_MODELS_MAP.get(dataset)
    if models:
        return models

    else:
        raise ValueError('Could not find models for dataset {}'.format(dataset))


def default_multi_classification(x_train, y_train, x_test, y_test, classifiers):
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        list[dict]:


    """
    performance = []
    for clf, name in classifiers:
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return performance


def default_binary_classification(x_train, y_train, x_test, y_test, classifiers):
    performance = []
    for clf, name in classifiers:
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='binary')

        performance.append(
            {
                "name": name,
                "accuracy": acc,
                "f1": f1
            }
        )

    return performance


def news_regression(x_train, y_train, x_test, y_test, regressors):
    performance = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    for clf, name in regressors:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        r2 = r2_score(y_test, pred)

        performance.append(
            {
                "name": name,
                "r2": r2,
            }
        )

    return performance


def make_features(data, meta, label_column='label', label_type='int', sample=50000):
    data = data.copy()
    np.random.shuffle(data)
    data = data[:sample]

    features = []
    labels = []

    for row in data:
        feature = []
        label = None
        for col, cinfo in zip(row, meta):
            if cinfo['name'] == 'label':
                if label_type == 'int':
                    label = int(col)
                elif label_type == 'float':
                    label = float(col)
                else:
                    assert 0, 'unkown label type'
                continue
            if cinfo['type'] == CONTINUOUS:
                if cinfo['min'] >= 0 and cinfo['max'] >= 1e3:
                    feature.append(np.log(max(col, 1e-2)))

                else:
                    feature.append((col - cinfo['min']) / (cinfo['max'] - cinfo['min']) * 5)

            elif cinfo['type'] == ORDINAL:
                feature.append(col)

            else:
                if cinfo['size'] <= 2:
                    feature.append(col)

                else:
                    tmp = [0] * cinfo['size']
                    tmp[int(col)] = 1
                    feature += tmp
        features.append(feature)
        labels.append(label)

    return features, labels


def default_gmm_likelihood(trainset, testset, n):
    gmm = GaussianMixture(n, covariance_type='diag')
    gmm.fit(testset)
    l1 = gmm.score(trainset)

    gmm.fit(trainset)
    l2 = gmm.score(testset)

    return [{
        "name": "default",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }]


def mapper(data, meta):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(meta):
            row_t.append(info['i2s'][int(row[id_])])
        data_t.append(row_t)
    return data_t


def default_bayesian_likelihood(dataset, trainset, testset, meta):
    struct = glob.glob("data/*/{}_structure.json".format(dataset))
    assert len(struct) == 1
    bn1 = BayesianNetwork.from_json(struct[0])

    trainset_mapped = mapper(trainset, meta)
    testset_mapped = mapper(testset, meta)
    prob = []
    for item in trainset_mapped:
        try:
            prob.append(bn1.probability(item))
        except Exception:
            prob.append(1e-8)

    l1 = np.mean(np.log(np.asarray(prob) + 1e-8))

    bn2 = BayesianNetwork.from_structure(trainset_mapped, bn1.structure)
    prob = []

    for item in testset_mapped:
        try:
            prob.append(bn2.probability(item))
        except Exception:
            prob.append(1e-8)

    l2 = np.mean(np.log(np.asarray(prob) + 1e-8))

    return [{
        "name": "default",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }]


DATASET_EVALUATOR_MAP = {
    "mnist12": default_multi_classification,
    "mnist28": default_multi_classification,
    "covtype": default_multi_classification,
    "intrusion": default_multi_classification,
    'credit': default_binary_classification,
    'census': default_binary_classification,
    'adult': default_binary_classification,
    'news': news_regression,
    'grid': default_gmm_likelihood,
    'gridr': default_gmm_likelihood,
    'ring': default_gmm_likelihood,
    'asia': default_bayesian_likelihood,
    'alarm': default_bayesian_likelihood,
    'child': default_bayesian_likelihood,
    'insurance': default_bayesian_likelihood,
}


def evaluate_dataset(dataset, trainset, testset, meta):
    # TODO: Use categoricals and ordinals instead of meta
    evaluator = DATASET_EVALUATOR_MAP.get(dataset)

    if evaluator is None:
        logging.warning("{} evaluation not defined.".format(dataset))
        return

    if dataset in ['asia', 'alarm', 'child', 'insurance']:
        return evaluator(dataset, trainset, testset, meta)

    if dataset in [
            "mnist12", "mnist28", "covtype", "intrusion", 'credit', 'census', 'adult', 'news']:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)
        return evaluator(x_train, y_train, x_test, y_test, get_models(dataset))

    bayesian_parameter = BAYESIAN_PARAMETER.get(dataset)
    if bayesian_parameter:
        return evaluator(trainset, testset, bayesian_parameter)


def compute_distance(trainset, syn, meta, sample=300):
    # TODO: Use categoricals and ordinals instead of meta
    mask_d = np.zeros(len(meta))

    for id_, info in enumerate(meta):
        if info['type'] in [CATEGORICAL, ORDINAL]:
            mask_d[id_] = 1
        else:
            mask_d[id_] = 0

    std = np.std(trainset, axis=0) + 1e-6

    dis_all = []
    for i in range(min(sample, len(trainset))):
        current = syn[i]
        distance_d = (trainset - current) * mask_d > 0
        distance_d = np.sum(distance_d, axis=1)

        distance_c = (trainset - current) * (1 - mask_d) / 2 / std
        distance_c = np.sum(distance_c ** 2, axis=1)
        distance = np.sqrt(np.min(distance_c + distance_d))
        dis_all.append(distance)

    return np.mean(dis_all)


def get_metadata(data, categoricals, ordinals, label):
    meta = []

    df = pd.DataFrame(data)
    for index in df:
        column = data[index]

        if index in categoricals:
            mapper = column.value_counts().index.tolist()
            meta.append({
                "name": index,
                "type": CATEGORICAL,
                "size": len(mapper),
                "i2s": mapper
            })
        elif index in ordinals:
            value_count = list(dict(column.value_counts()).items())
            value_count = sorted(value_count, key=lambda x: -x[1])
            mapper = list(map(lambda x: x[0], value_count))
            meta.append({
                "name": index,
                "type": ORDINAL,
                "size": len(mapper),
                "i2s": mapper
            })
        else:
            meta.append({
                "name": index,
                "type": CONTINUOUS,
                "min": column.min(),
                "max": column.max(),
            })

        if index == label:
            meta[-1]['name'] = 'label'

    return meta


def evaluate(train, test, synthesized_data, categoricals=None, ordinals=None, label=None):
    categoricals = categoricals or list()
    ordinals = ordinals or list()

    meta = get_metadata(train, categoricals, ordinals, label)

    results = []
    for step, synth_data in enumerate(synthesized_data):
        performance = evaluate_dataset('intrusion', synth_data, test, meta)
        distance = compute_distance(train, synth_data, meta)

        for perf in performance:
            perf['step'] = step
            perf['distance'] = distance
            results.append(perf)

    return results
