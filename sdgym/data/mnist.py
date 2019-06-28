
import json
import os

import numpy as np

import cv2
from keras.datasets import mnist
from sdgym.utils import CATEGORICAL, verify

output_dir = "data/real/"
temp_dir = "tmp/"


def make_data(t_train, t_test, wh, name):
    np.random.seed(0)

    assert t_train.shape[1] == wh * wh + 1
    assert t_test.shape[1] == wh * wh + 1

    meta = []
    for i in range(wh):
        for j in range(wh):
            meta.append({
                "name": "%02d%02d" % (i, j),
                "type": CATEGORICAL,
                "size": 2,
                "i2s": ["0", "1"]
            })
    meta.append({
        "name": "label",
        "type": CATEGORICAL,
        "size": 10,
        "i2s": [str(x) for x in range(10)]
    })

    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))

    np.random.shuffle(t_train)

    t_train = t_train.astype('int8')
    t_test = t_test.astype('int8')

    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)
    verify("{}/{}.npz".format(output_dir, name), "{}/{}.json".format(output_dir, name))

    # Sample
    for i in range(5):
        img = t_train[i][:-1].reshape([wh, wh]) * 255
        lb = t_train[i][-1]
        cv2.imwrite('{}/{}_{}_{}.png'.format(temp_dir, name, i, lb), img)


if __name__ == "__main__":
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    t_train = np.concatenate(
        [
            (x_train > 128).astype('int32').reshape([60000, -1]),
            y_train.reshape([60000, -1])
        ],
        axis=1
    )

    t_test = np.concatenate(
        [
            (x_test > 128).astype('int32').reshape([10000, -1]),
            y_test.reshape([10000, -1])
        ],
        axis=1
    )

    make_data(t_train, t_test, 28, 'mnist28')

    x_train_r = np.asarray([cv2.resize(im, (12, 12)) for im in x_train])
    x_test_r = np.asarray([cv2.resize(im, (12, 12)) for im in x_test])

    t_train_r = np.concatenate(
        [
            (x_train_r > 128).astype('int32').reshape([60000, -1]),
            y_train.reshape([60000, -1])
        ],
        axis=1
    )

    t_test_r = np.concatenate(
        [
            (x_test_r > 128).astype('int32').reshape([10000, -1]),
            y_test.reshape([10000, -1])
        ],
        axis=1
    )

    make_data(t_train_r, t_test_r, 12, 'mnist12')