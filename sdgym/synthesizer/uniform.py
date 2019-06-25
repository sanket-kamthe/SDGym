import numpy as np

from sdgym.synthesizers.base import SynthesizerBase, run


rng = np.random


class UniformSynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""

    def train(self, train_data):
        self.dtype = train_data.dtype
        self.shape = train_data.shape

    def generate(self, n):
        data = rng.uniform(0, 1, (n, self.shape[1]))

        for i, c in enumerate(self.meta):
            if c['type'] == 'continuous':
                data[:, i] = data[:, i] * (c['max'] - c['min']) + c['min']
            else:
                data[:, i] = (data[:, i] * (1-1e-8) * c['size']).astype('int32')

        return [(0, data.astype(self.dtype))]
