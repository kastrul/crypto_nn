import numpy as np


class PastSampler:
    """
    Forms training samples for predicting future values from past value
    """

    def __init__(self, past_samples, future_samples):
        """
        Predict 'fs' future sample using 'ps previous samples
        """
        self.fs = future_samples
        self.ps = past_samples

    def transform(self, stacked_data, Y=None):
        total_samples = self.ps + self.fs  # Number of samples per row (sample + target)
        # Matrix of sample indices like: {{1, 2..., M}, {2, 3, ..., M + 1}}
        indices = np.arange(total_samples) + np.arange(stacked_data.shape[0] - total_samples + 1).reshape(-1, 1)
        shaped_data = stacked_data[indices].reshape(-1, total_samples * stacked_data.shape[1], *stacked_data.shape[2:])
        ci = self.ps * stacked_data.shape[1]  # Number of features per sample
        return shaped_data[:, :ci], shaped_data[:, ci:]  # Sample matrix, Target matrix
