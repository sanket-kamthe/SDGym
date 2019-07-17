import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sdgym.evaluate import evaluate

DATASETS = {
    'iris': {
        'label': 4
    },
    # 'adult': {
    #     'categoricals': [
    #     ],
    #     'ordinals': [
    #     ]
    # },
}


def load_dataset(name):
    # TODO: Make this real
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=0)
    train = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    test = np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1)

    return train, test


def benchmark(synthesizer, datasets=DATASETS, repeat=3):

    results = list()
    for name, values in datasets.items():
        train, test = load_dataset(name)

        iteration_results = list()
        for iteration in range(repeat):
            synthesized = [
                synthesizer(
                    train,
                    values.get('categorical', list()),
                    values.get('ordinal', list())
                )
                for _ in range(repeat)
            ]

            dataset_results = evaluate(train, test, synthesized, **values)
            dataset_results = pd.DataFrame(dataset_results)
            dataset_results['dataset'] = name
            dataset_results['iter'] = iteration
            iteration_results.append(dataset_results)

        results.extend(iteration_results)

    return pd.concat(results)
