import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Evaluate output of one synthesizer.')
parser.add_argument('--result', type=str, default='output/__result__',
                    help='result dir')
parser.add_argument('--summary', type=str, default='output/__summary__',
                    help='result dir')


METHOD_ORDER = dict(
    identity=0,
    clbn=1,
    privbn=2,
    medgan=3,
    veegan=4,
    tablegan=5,
    tvae=8,
    tgan=9
)


def method_name_order(item):
    return METHOD_ORDER.get(item[0].lower(), 6)


def coverage(datasets, results):
    ticks = []
    values = []

    results = list(results)
    results = sorted(results, key=method_name_order)

    for model, result in results:
        covered = set()
        for item in result:
            assert(item['dataset'] in datasets)
            covered.add(item['dataset'])

        ticks.append(model)
        values.append(len(covered) / len(datasets))

    plt.cla()
    plt.bar(list(range(len(values))), values, tick_label=ticks)
    plt.xticks(rotation=-90)
    plt.title("coverage")
    plt.ylim(0, 1)

    plt.savefig("{}/coverage.jpg".format(summary_dir), bbox_inches='tight')


def save_barchart(barchart, filename):
    barchart = pd.DataFrame(barchart, columns=['synthesizer', 'metric', 'val'])

    methods = set()
    for item in barchart['synthesizer']:
        methods.add(item)
    methods = list(methods)
    methods = sorted(methods, key=method_name_order)

    barchart.pivot("metric", "synthesizer", "val")[methods].plot(kind='bar')
    plt.title(dataset)
    plt.xlabel(None)
    plt.legend(title=None, loc=(1.04, 0))
    plt.savefig(filename, bbox_inches='tight')


def dataset_performance(dataset, results):
    synthesizer_metric_perform = {}

    for synthesizer, all_result in results:
        for one_result in all_result:
            if one_result['dataset'] != dataset:
                continue

            if one_result['step'] == 0:
                synthesizer_name = synthesizer
            else:
                synthesizer_name = synthesizer + "_" + str(one_result['step'])

            if synthesizer_name not in synthesizer_metric_perform:
                distance = [one_result.get('distance', 0)]
                synthesizer_metric_perform[synthesizer_name] = {
                    '_distance': distance
                }

            for model_metric_score in one_result['performance']:
                for metric, v in model_metric_score.items():
                    if metric == "name":
                        continue
                    else:
                        if metric not in synthesizer_metric_perform[synthesizer_name]:
                            synthesizer_metric_perform[synthesizer_name][metric] = []

                        synthesizer_metric_perform[synthesizer_name][metric].append(v)

    if len(synthesizer_metric_perform) == 0:
        return

    plt.cla()

    barchart = []
    barchart_d = []
    for synthesizer, metric_perform in synthesizer_metric_perform.items():
        for k, v in metric_perform.items():
            v_t = np.mean(v)
            if k == 'r2':
                v_t = v_t.clip(-1, 1)
            if 'likelihood' in k:
                v_t = v_t.clip(-20, 0)
            if k == '_distance':
                barchart_d.append((synthesizer, k, v_t))
            else:
                barchart.append((synthesizer, k, v_t))

    save_barchart(barchart, "{}/{}.jpg".format(summary_dir, dataset))
    save_barchart(barchart_d, "{}/{}_d.jpg".format(summary_dir, dataset))

    return synthesizer_metric_perform


def generate_tabular_result(dataset_perform):
    df = pd.DataFrame(data={'alg': []})

    for dataset, alg_metric_perform in dataset_perform.items():
        for alg, metric_perform in alg_metric_perform.items():
            for metric, perform in metric_perform.items():
                column_name = "{}/{}".format(dataset, metric)
                row_name = alg

                if column_name not in df.columns:
                    df[column_name] = [None] * len(df)

                if row_name not in set(df['alg'].unique()):
                    row_id = len(df)
                    df.loc[row_id] = [None] * len(df.columns)
                    df['alg'][row_id] = alg
                else:
                    row_id = df[df['alg'] == row_name].index[0]

                df[column_name][row_id] = np.mean(perform)

    df.to_csv("{}/results.csv".format(summary_dir))


if __name__ == "__main__":
    args = parser.parse_args()

    result_files = glob.glob("{}/*.json".format(args.result))
    summary_dir = args.summary

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    datasets = glob.glob("data/*/*.npz")
    datasets = [re.search('.*/([^/]*).npz', item).group(1) for item in datasets]

    results = []
    for result_file in result_files:
        model = re.search('.*/([^/]*).json', result_file).group(1)
        with open(result_file) as f:
            res = json.load(f)

        results.append((model, res))

    coverage(datasets, results)

    dataset_perform = {}
    for dataset in datasets:
        perform = dataset_performance(dataset, results)
        if perform is None:
            continue
        else:
            dataset_perform[dataset] = perform

    generate_tabular_result(dataset_perform)
