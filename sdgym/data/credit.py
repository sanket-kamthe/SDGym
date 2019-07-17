# Generate credit datasets

import json
import os

import numpy as np
import pandas as pd

from sdgym.synthesizers.utils import CATEGORICAL, CONTINUOUS


def process_credit_dataset(path, output_dir="data/real/"):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    df = pd.read_csv(path)
    df.drop(columns=['Time'], inplace=True)
    values = df.values

    meta = []
    for i in range(28):
        meta.append({
            "name": "V%d" % i,
            "type": CONTINUOUS,
            "min": np.min(values[:, i]),
            "max": np.max(values[:, i])
        })
    meta.append({
        "name": "Amount",
        "type": CONTINUOUS,
        "min": np.min(values[:, 28]),
        "max": np.max(values[:, 28])
    })
    meta.append({
        "name": "label",
        "type": CATEGORICAL,
        "size": 2,
        "i2s": ["0", "1"]
    })

    np.random.seed(0)
    np.random.shuffle(values)
    t_train = values[:-20000].astype('float32')
    t_test = values[-20000:].astype('float32')

    name = "credit"
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))

    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)
