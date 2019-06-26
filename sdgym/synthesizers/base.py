class BaseSynthesizer:
    """Base class for all default synthesizers of ``SDGym``."""

    def __call__(self, data):
        """Call an instance like a function.

        This magic method is what allows the duality class-function, as the objects are defined as
        classes, but executed as function.

        Args:
            data(pandas.DataFrame): Table of data to synthesize.

        Returns:
            pandas.DataFrame:
                Synthesized data. It will contain the same number of rows, columns and index

        """
        self.fit(data)
        return self.sample(data.shape[0])

    def fit(self, data):
        """Model and prepare instance to generate samples.

        Args:
            data(pandas.DataFrame): Data to model.

        Returns:
            None

        """
        raise NotImplementedError

    def sample(self, num_samples):
        """Generate new samples.

        Args:
            num_samples(int): Number of samples to generate.

        Returns:
            pandas.DataFrame: Synthesized data.

        """
        raise NotImplementedError
