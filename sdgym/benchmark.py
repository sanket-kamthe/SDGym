

SYNTHESIZERS = []
DATASETS = []


def synthesize(synthesizer, datasets):
    """Synthesize all datasets with synthesizer.

    Args:
        synthesizer(callable):
            A synthesizer function.
        datasets(dict[str, tuple]):
            The tuple 
            - The name of the dataset.
            - The actual data as an np.ndarray.
            - A list of categorical columns.
            - A list of ordinal columns.

    Return:
        dict[str, tuple]:
            A dict whose values are tuples of real and synthesized tables and their annotations.
    """
    result = {}
    for name, (real_data, categorical_columns, ordinal_columns in datasets:
        synthetic_data = synthesizer(real_data, categorical_columns, ordinal_columns)
        result[name] = (real_data, synthetic_data)

    return result


def evaluate(synthesized_datasets):
    """Evaluate a set of synthesized results.

    Args:
        synthesized_datasets(dict[str, tuple]):

    Returns:

    """
    pass


def evalute_dataset(dataset, trainset, testset, meta):

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

def benchmark(synthesizer, baseline=SYNTHESIZERS, datasets=DATASETS, repetitions=5):
    """ """
    synthezised
    for name, real_data, continous_columns, ordinal_columns in datasets:
        synthetic_data = synthesizer(real_data, continous_columns, ordinal_columns)
