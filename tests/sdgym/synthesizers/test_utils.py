from unittest import TestCase

import pandas as pd

from sdgym.synthesizers.utils import (
    BGMTransformer, DiscretizeTransformer, GeneralTransformer, GMMTransformer, Transformer)


class TestTransformer(TestCase):

    def test_get_metadata(self):
        """get_metadata returns information about the dataframe."""
        # Setup
        data = pd.DataFrame({
            'numerical': [0, 1, 2, 3, 4, 5],
            'categorical': list('AAABBC')
        })
        data['categorical'] = data.categorical.astype('category')

        expected_result = [
            {
                'name': 'numerical',
                'type': 'continuous',
                'min': 0,
                'max': 5
            },
            {
                "name": 'categorical',
                "type": 'categorical',
                "size": 3,
                "i2s": ['A', 'B', 'C']
            }
        ]

        # Run
        result = Transformer.get_metadata(data)

        # Check
        assert result == expected_result


class TestDiscretizeTransformer(TestCase):

    def test_transform(self):
        """transform continous columns into discrete bins."""
        # Setup
        transformer = DiscretizeTransformer(n_bins=5)
        data = pd.DataFrame({
            'A': [x for x in range(100)],
            'B': [2 * x for x in range(100)]
        })
        binned_column = ([0] * 20) + ([1] * 20) + ([2] * 20) + ([3] * 20) + ([4] * 20)
        expected_result = pd.DataFrame({
            'A': binned_column,
            'B': binned_column
        })
        transformer.fit(data)

        # Run
        result = transformer.transform(data)

        # Check
        assert result.equals(expected_result)
