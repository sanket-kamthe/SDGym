<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SDGym” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>



[![Travis](https://travis-ci.org/DAI-Lab/SDGym.svg?branch=master)](https://travis-ci.org/DAI-Lab/SDGym)
[![PyPi Shield](https://img.shields.io/pypi/v/sdgym.svg)](https://pypi.python.org/pypi/sdgym)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/SDGym/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/SDGym)
[![Downloads](https://pepy.tech/badge/sdgym)](https://pepy.tech/project/sdgym)


# SDGym - Synthetic Data Gym

- License: MIT
- Documentation: https://DAI-Lab.github.io/SDGym/
- Homepage: https://github.com/DAI-Lab/SDGym

# Overview

Synthetic Data Gym (SDGym) is a framework to benchmark the performance of synthetic data generators
for non-temporal tabular data. SDGym is based on a [paper](??)
of the same name, and the project is part of the [Data to AI Laboratory](https://dai.lids.mit.edu/) at MIT.


# Install

## Requirements


**SDGym** has been developed and tested on [Python 3.5, and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **SDGym** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **SDGym**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) sdgym-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source sdgym-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **SDGym**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **SDGym**:

```bash
pip install sdgym
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/SDGym.git
cd SDGym
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

First, please head to [the GitHub page of the project](https://github.com/DAI-Lab/SDGym)
and make a fork of the project under you own username by clicking on the **fork** button on the
upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes
the number of the issue that you are going to work on:

```bash
git clone git@github.com:{your username}/SDGym.git
cd SDGym
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, which will install some additional
dependencies for code linting and testing.

```bash
make install-develop
```

Make sure to use them regularly while developing by running the commands `make lint` and `make test`.


# Input Format

## Data Format

All datasets used in this project, either provided by **SDGym** or by the user will should have
the following characteristics:

* It must be a `pandas.DataFrame` object
* All categorical values have to use the
  [categorical dtype](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).

Also, when loading a custom dataset from a CSV file, it .

* It uses a single comma, `,`, as the separator.
* Its first row is a header that contains the names of the columns.
* It contains no index column

### Demo Datasets

`SDGym` includes a few datasets to use for development or demonstration purposes. These datasets
have been preprocessed to be ready to use with `SDGym`, following the requirements specified in
the [Input Format](#input-format) section.

These datasets can be downloaded from [here](https://s3.amazonaws.com/sdgym/SDGymBenchmarkData.zip).
After downloading them, you just need to unzip their contents into a folder named `data` at the
root of `SDGym`.

You can also execute the following commands from the root of the repository:
```
curl https://s3.amazonaws.com/sdgym/SDGymBenchmarkData.zip -o data.zip
mkdir data
unzip data.zip -d data/
```

Have below the list of included datasets and their original source:

- MINIST28: Use flatten 28\*28 pixels into 784 binary columns with an extra label column.
- MINIST12: Reshape 28\*28 pixels into 12\*12 binary columns with an extra label column.
- Credit: Kaggle credit card fraud. [Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Adult: Adult Dataset. [Link](https://archive.ics.uci.edu/ml/datasets/adult)
- Census: KDD Census dataset.  [Link](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD))
- News: Online News Popularity Dataset (Regression).  [Link](https://archive.ics.uci.edu/ml/datasets/online+news+popularity)
- Covertype: Covertype Dataset (8 continuous + 40 binary + 1 multi).  [Link](https://archive.ics.uci.edu/ml/datasets/Covertype)
- Intrusion: Network intrusion detector kdd99.  [Link](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data)


## Model Format

Considering that there can be multiple possible API that models can implement and is virtually
impossible to consider all of them, the approach this project has taken is to accept as
synthesizers only functions that takes as arguments a `pandas.DataFrame` with the format mentioned
in the [Data Format](#data-format) section.

So in order to benchmark a model using **SDGym**, you only need to wrap in a function that takes
as only argument a `pandas.DataFrame` and returns a synthesized one, with the same number of rows
as the original.


# Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with **SDGym** by exploring its Python API.


## 1. Prepare your model

As we mentioned on the [Model Format](#Model-Format) section above, in order to benchmark it, we
must wrap our model in a function that takes as argument a table of data and returns a table of
synthesized data.

All parameters required to instantiate, fit and sample your model must be **inside** of the function.

```python
from my_package.my_module import MyModel

def synthesizer(X):
    num_rows = X.shape[0]

    model = MyModel()
    model.fit(X)

    return model.sample(num_rows)
```

# 2. Run

Now we can run the `sdgym.benchmark` function to test our model:

```python
from sdgym import benchmark

score = benchmark(synthesizer)
```

# What's next?

For more details about **SDGym** and all its possibilities and features, please check the
[documentation site](https://DAI-Lab.github.io/SDGym/).

There you can learn more about [how to contribute to SDGym](https://HDI-Project.github.io/SDGym/community/contributing.html)
in order to help us developing new features or cool ideas.

# Credits

SDGym is an open source project from the Data to AI Lab at MIT which has been built and maintained
over the years by the following team:

* Lei Xu <leonard.xu.thu@gmail.com>
* Kalyan Veeramachaneni <kalyan@csail.mit.edu>
* Manuel Alvarez <manuel@pythiac.com>


## Citing SDGym


## Related Projects

### SDV

[SDV](https://github.com/HDI-Project/SDV), for Synthetic Data Vault, is the end-user library for
synthesizing data in development under the [HDI Project](https://hdi-dai.lids.mit.edu/).
SDV allows you to easily model and sample relational datasets using Copulas thought a simple API.
Other features include anonymization of Personal Identifiable Information (PII) and preserving
relational integrity on sampled records.

### TGAN

[TGAN](https://github.com/DAI-Lab/TGAN) is a GAN based model for synthesizing tabular data.
It's also developed by the [MIT's Data to AI Lab](https://dai-lab.github.io/) and is under
active development.
