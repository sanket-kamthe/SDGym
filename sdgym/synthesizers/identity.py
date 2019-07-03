import numpy as np

from sdgym.synthesizers.base import BaseSynthesizer


class IdentitySynthesizer(BaseSynthesizer):
    """Trivial synthesizer.

    Returns the same exact data that is used to fit it.
    """

    def fit(self, train_data):
        self.data = train_data.copy()

    def sample(self, num_samples):
        assert self.data.shape[0] >= num_samples
        np.random.shuffle(self.data)
        return [(0, self.data[:num_samples])]
