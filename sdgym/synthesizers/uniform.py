import numpy as np

from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import Transformer


class UniformSynthesizer(BaseSynthesizer):
    """UniformSynthesizer."""

    def fit(self, data):
        self.dtype = data.dtype
        self.shape = data.shape
        self.meta = Transformer.get_metadata(data, self.categoricals, self.ordinals)

    def sample(self, n):
        data = np.random.uniform(0, 1, (n, self.shape[1]))

        for i, c in enumerate(self.meta):
            if c['type'] == 'continuous':
                data[:, i] = data[:, i] * (c['max'] - c['min']) + c['min']
            else:
                data[:, i] = (data[:, i] * (1 - 1e-8) * c['size']).astype('int32')

        return data.astype(self.dtype)
