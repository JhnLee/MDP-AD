import numpy as np
from itertools import islice
from collections import Counter, OrderedDict


class MdpAD:
    def __init__(self, transition_dim=2, input_data=None, threshold=None):
        '''
        Each log key value is added by 3, and then assigned to each col(row) of Transition matrix 
        --------------
        BOS key -> 0
        EOS key -> 1
        Unknown key -> 2
        
        (EXAMPLE) 2, 1, 3 ... -> 0, 5, 4, 6 ...
        (EXAMPLE) 6, -1, 2 ... -> 0, 9, 2, 5 ...
        '''
        if type(transition_dim) is not int:
            raise ValueError('transition dimension should be integer format')
        if transition_dim <= 1:
            raise ValueError('dimension should be bigger than 1')
        
        self.dim = transition_dim
        self.threshold = threshold
        self.BOS_key, self.EOS_key, self.UNK_key = 0, 1, 2
        self.training_data = None
        
        if input_data is not None:
            if type(input_data) is not dict:
                raise ValueError('Input data is not in a valid format (Should be a dictionary)')            
            self.training_data = input_data

    def fit(self, input_data=None):
        '''Fit Anomaly Detector to training dataset'''
        if input_data is None:
            if self.training_data is None:
                raise ValueError('No input data')
            else: 
                self.training_data =self._encode(input_data)
        else:
            if self.training_data is not None:
                raise ValueError('Got 2 input data')
            else:
                self.training_data = self._encode(input_data)

        self._make_transition_matrix()

    def predict(self, test_data):
        '''Predict novelty score from test dataset'''
        if type(test_data) is not dict:
            raise ValueError('Input data is not in a valid format (Should be a dictionary)')

        test_data = self._encode(test_data)
        score = {k: self._get_novelty_score(s) for k, s in test_data.items()}
        if self.threshold is None:
            return score
        
        # Return anomaly score whose score is higher than the threshold
        anomaly = {k: s for k, s in score if s > self.threshold}
        return anomaly

    def _make_transition_matrix(self):
        '''Make transition matrix from whole log sequences'''
        count = Counter()
        for d in self.training_data.values():
            count.update(d)
        
        # Add idx of UNK token
        count[2] = 1
        
        # Sort counter by key
        count = dict(sorted(count.items(), key=lambda x: x[0]))
        n_log = list(count.keys())[-1] + 1
        
        trans_mat = np.zeros(tuple([n_log] * self.dim))
        window_sequences = [self._window(x) for x in list(self.training_data.values())]

        # Update frequency on the transition matrix
        for seq in window_sequences:
            for window in seq:
                trans_mat[window] += 1

        # Make frequency to prob
        log_count = np.array(list(count.values()))[:, np.newaxis]
        if self.dim > 2:
            # prob = P(X == x_i | x_i-1, x_i-2, ...)
            # --> log count = P(x_i-1, x_i-2, ...)
            log_count = np.expand_dims(np.sum(trans_mat, axis=self.dim-1), axis=self.dim-1)
            # to avoid error about dividing zero
            log_count = np.where(log_count > 0, log_count, 1)
        
        self.transition_matrix = trans_mat / log_count       
        
        count[2] = 0
        self.training_data_count = count
        
        # Decode the log sequences
        self.training_data = self._decode(self.training_data)

    def _get_novelty_score(self, seq):
        '''Calculate expectation of transition prob in the log sequence'''
        if type(seq) is not list:
            raise ValueError('Input data should be in a list format')
        windows = list(self._window(seq))

        def get_neg_log(x):
            prob = self.transition_matrix[x]
            return -np.log(1e-9 if prob == 0 else prob)

        neg_log_sum = np.sum(list(map(get_neg_log, windows)))
        return neg_log_sum / (len(seq) - (self.dim-1))

    def _window(self, seq):
        '''Make sequence of transition from log sequence'''
        # BOS  3  4  5  EOS  =>  (BOS, 3)  (3, 4)  (4, 5)  (5, EOS)
        it = iter(seq)
        result = tuple(islice(it, self.dim))
        if len(result) == self.dim:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
            
    def _encode(self, data):
        '''Add 3 to every element in the sequence'''
        data = OrderedDict([(b, [self.BOS_key] + [e + 3 for e in s] + [self.EOS_key])
                          for b, s in data.items()])
        return data
    
    def _decode(self, data):
        '''Subtract 3 from every key and delete EOS and BOS tokens'''
        keys = list(data.keys())
        vals = list(data.values())
        vals = [val[1:-1] for val in vals]
        data = OrderedDict([(key, [e - 3 for e in val]) 
                            for key, val in zip(keys, vals)])
        return data