import numpy as np
from itertools import islice


class MdpAD:
    def __init__(self, input_data=None):
        self.data = None
        if input_data is not None:
            self._initialize_seq(input_data)

    def fit(self, input_data=None):

        if input_data is None:
            if self.data is None:
                raise ValueError('No input data')

        else:
            if self.data is not None:
                raise ValueError('Got 2 input data')
            else:
                self._initialize_seq(input_data)

        self.make_transition_matrix()

    def predict(self, test_data, threshold=None):
        if type(test_data) is not dict:
            raise ValueError('Input data is not in a valid format (Should be a dictionary)')

        test_data = dict([(b, [self.BOS_key] + s + [self.EOS_key])
                          for b, s in test_data.items()])

        score = {k: self.get_novelty_score(s) for k, s in test_data.items()}

        if threshold is None:
            return score

        anomaly = dict([(i, s) for i, s in score if s < threshold])
        return anomaly

    def make_transition_matrix(self):
        unlisted_log = [x for y in list(self.data.values()) for x in y]
        unique_logs, log_count = np.unique(unlisted_log, return_counts=True)
        self.data_counter = dict(zip(unique_logs, log_count))

        n_log = len(unique_logs)
        self.transition_matrix = np.zeros((n_log, n_log))

        window_sequences = [self._window(x) for x in list(self.data.values())]

        # Update frequency on transition matrix
        self._update_transition_freq(window_sequences)

        # Make frequency to prob
        self.transition_matrix /= log_count[:, np.newaxis]

    def get_novelty_score(self, seq):
        if type(seq) is not list:
            raise ValueError('Input data should be in a list format')
        windows = list(self._window(seq))

        def get_neg_log(x):
            # Return 0 probability if window contains UNK token
            prob = self.transition_matrix[x]
            if -1 in x or prob == 0:
                prob = 1e-9

            return -np.log(prob)

        neg_log_sum = np.sum(list(map(get_neg_log, windows)))
        n = len(seq)
        return neg_log_sum / (n - 1)

    def _initialize_seq(self, input_data):
        if type(input_data) is not dict:
            raise ValueError('Input data is not in a valid format (Should be a dictionary)')

        self.unique_logs = self._get_unique_log(input_data)
        self.BOS_key = len(self.unique_logs)
        self.EOS_key = len(self.unique_logs) + 1
        self.data = dict([(b, [self.BOS_key] + s + [self.EOS_key])
                          for b, s in input_data.items()])
        self.data_id = input_data.keys()

    def _get_unique_log(self, seq):
        unlisted_log = [x for y in list(seq.values()) for x in y]
        return np.unique(unlisted_log)

    def _window(self, seq, n=2):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def _update_transition_freq(self, data):
        for seq_window in data:
            for window in seq_window:
                self.transition_matrix[window] += 1