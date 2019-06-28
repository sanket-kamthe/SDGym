import glob
import json

import numpy as np
import pandas as pd


def process_dataset(dataset, dataset_type, summary):
    summary["type"].append(dataset_type)
    summary['name'].append(dataset.split('/')[-1][:-4])

    data = np.load(dataset)
    with open(dataset[:-3] + 'json') as f:
        meta = json.load(f)

    summary['#train'].append(len(data['train']))
    summary['#test'].append(len(data['test']))
    summary['#column'].append(len(meta))
    summary['#continuous'].append(
        len([0 for info in meta if info['type'] == 'continuous']))
    summary['#ordinal'].append(
        len([0 for info in meta if info['type'] == 'ordinal']))
    summary['#binary'].append(
        len([0 for info in meta if info['type'] == 'categorical' and info['size'] <= 2]))
    summary['#multi'].append(
        len([0 for info in meta if info['type'] == 'categorical' and info['size'] > 2]))
    summary['task'].append(
        'likelihood' if dataset_type == 'simulated' else 'classification'
        if 'news' not in dataset else 'regression')


if __name__ == "__main__":
    datasets = glob.glob("data/simulated/*.npz")
    datasets = sorted(datasets)
    summary = {
        "type": [],
        "name": [],
        "#train": [],
        "#test": [],
        "#column": [],
        "#continuous": [],
        "#ordinal": [],
        "#binary": [],
        "#multi": [],
        "task": []
    }

    for dataset in datasets:
        process_dataset(dataset, "simulated")

    datasets = glob.glob("data/real/*.npz")
    datasets = sorted(datasets)
    for dataset in datasets:
        process_dataset(dataset, "real")

    df = pd.DataFrame(summary)
    df.to_csv('dataset.csv', index=None)