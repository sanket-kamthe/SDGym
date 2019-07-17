import numpy as np
from sklearn.mixture import GaussianMixture

from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import Transformer
from sdgym.utils import CONTINUOUS

rng = np.random


class IndependentSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, categoricals, ordinals, gmm_n=5):
        self.gmm_n = gmm_n
        super().__init__(categoricals, ordinals)

    def fit(self, data):
        self.dtype = data.dtype
        self.meta = Transformer.get_metadata(data, self.categoricals, self.ordinals)

        self.models = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                model = GaussianMixture(self.gmm_n)
                model.fit(data[:, [id_]])
                self.models.append(model)
            else:
                nomial = np.bincount(data[:, id_].astype('int'), minlength=info['size'])
                nomial = nomial / np.sum(nomial)
                self.models.append(nomial)

    def sample(self, n):
        data = np.zeros([n, len(self.meta)], self.dtype)

        for i, info in enumerate(self.meta):
            if info['type'] == 'continuous':
                x, _ = self.models[i].sample(n)
                rng.shuffle(x)
                data[:, i] = x.reshape([n])
                data[:, i] = data[:, i].clip(info['min'], info['max'])
            else:
                data[:, i] = np.random.choice(np.arange(info['size']), n, p=self.models[i])

        return data
