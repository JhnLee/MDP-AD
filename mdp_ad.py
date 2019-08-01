import numpy as np
import pandas as pd
import config
import pickle
from itertools import islice
from collections import Counter, OrderedDict


class MdpAD:
    def __init__(self, window_size=2, input_data=None, threshold=None):
        '''
        Each log key value is added by 2, and then assigned to each col(row) of Transition matrix 
        --------------
        BOS key -> 0
        EOS key -> 1        
        
        (EXAMPLE) 2, 1, 3 ... -> 0, 5, 4, 6 ...
        (EXAMPLE) 6, 12, 2 ... -> 0, 9, 2, 5 ...
        
        Params
        --------------
        window_size : Window size of Log key sequence.
            This would be the number of elements joint probabilties contain
        
        threshold : If True, predict function only return sequences
            whose novelty score is upper than the threshold
        '''
        if type(window_size) is not int:
            raise ValueError('window_size should be integer format, your file format is {}'.format(type(window_size)))
        if window_size <= 1:
            raise ValueError('window_size should be bigger than 1')
        
        self.dim = window_size
        self.threshold = threshold
        self.BOS_key, self.EOS_key = 0, 1
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
                self.training_data =self._encode(self.training_data)
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
        
        # Exclude unseen log key which is assigned as -1 by Drain parser
        self.test_data = {k: v for k, v in test_data.items() if not -1 in v}
        unseen_num = np.sum([-1 in v for v in list(test_data.values())])
        if unseen_num > 0:
            print("Number of unseen log key during prediction: {}".format(unseen_num))
            print("Sequences having unseen log key are excluded")
        
        self.test_data = self._encode(self.test_data)
        score = {k: np.sum(self.get_seq_score(s)) / (len(s) - (self.dim-1)) 
                 for k, s in self.test_data.items()}
        self.test_data = self._decode(self.test_data)
        
        if self.threshold is None:
            return score
        
        # Return anomaly score whose score is higher than the threshold
        anomaly = {k: s for k, s in score if s > self.threshold}
        return anomaly

    def count_train_log(self):
        if self.training_data:
            count = Counter()
            for d in self.training_data.values():
                count.update(d)
            return dict(sorted(count.items(), key=lambda x: x[0]))
        else:
            raise ValueError('No training data exist')
    
    def _make_transition_matrix(self):
        '''Make transition matrix from whole log sequences'''
        # prob = P(X == x_i | x_i-1, x_i-2, ...)
        # --> denominator = P(x_i-1, x_i-2, ...)
        # --> numerator = P(x_i, x_i-1, x_i-2, ...)
        
        # size of transition matrix : (max log key value + 3)^(dim)        
        freq_mat = np.zeros(tuple([self._maximum_key + 3] * self.dim))
        window_sequences = [self._window(x) for x in list(self.training_data.values())]

        # Update frequency on the transition matrix
        for seq in window_sequences:
            for window in seq:
                freq_mat[window] += 1

        # Make frequency to prob      
        denom = np.expand_dims(np.sum(freq_mat, axis=self.dim-1), axis=self.dim-1)
        # Temprarily assign value 1 to avoid Devide by Zero error
        # [0, 0, 0, 0] / 0 => [0, 0, 0, 0] / 1
        denom = np.where(denom>0, denom, 1)
        self._transition_matrix = np.divide(freq_mat, denom)    

        # Decode the log sequences
        self.training_data = self._decode(self.training_data)
    
    @property
    def transition_matrix(self):
        return self._transition_matrix
    
    def get_seq_score(self, seq):
        '''Calculate transition prob of each window in the log sequence'''
        if type(seq) is not list:
            raise ValueError('Input data should be in a list format')
        windows = list(self._window(seq))
        
        def get_neg_log(x):
            prob = self._transition_matrix[x]
            return -np.log(1e-9 if prob == 0 else prob)
        
        return list(map(get_neg_log, windows))

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
        '''Add 2 to every element in the sequence and track maximum key value'''
        encoded_data = OrderedDict()
        maximum_key = -1
        
        for b, s in data.items():
            max_tmp = max(s)
            maximum_key = max_tmp if max_tmp > maximum_key else maximum_key
            encoded_data[b] = [self.BOS_key] + [e + 2 for e in s] + [self.EOS_key]
        
        self._maximum_key = maximum_key
        
        return encoded_data
    
    def _decode(self, data):
        '''Subtract 2 from every key and delete EOS and BOS tokens'''
        keys = list(data.keys())
        vals = list(data.values())
        vals = [val[1:-1] for val in vals]
        data = OrderedDict([(key, [e - 2 for e in val]) 
                            for key, val in zip(keys, vals)])

        return data

    
def main():
    with open(config.TRAIN_DATA_PATH, 'rb') as f:
        trn = pickle.load(f)
    with open(config.TEST_DATA_PATH, 'rb') as f:
        tst = pickle.load(f)
    
    anomaly_label = pd.read_csv(config.LABEL_PATH)
    id2label = dict(zip(anomaly_label['BlockId'], anomaly_label['Label']))
    
    print("Fitting on Training Dataset...")
    mdp = MdpAD(config.WINDOW_SIZE, trn, config.THRESHOLD)
    mdp.fit()
    print("Done.")
    
    # Save transition Matrix
    with open(config.TRANSITION_MATRIX_PATH, 'wb') as f:
        pickle.dump(mdp.transition_matrix, f)
    print('Saved Transition Matrix')
    
    mdp_pred = mdp.predict(tst)
    mdp_pred_score = list(mdp_pred.values())
    
    print("Saving result file...")
    result = pd.DataFrame({'BlockID': list(mdp.test_data.keys()),
                          'log_sequence': list(mdp.test_data.values()),
                          'window_neg_log_score': [mdp.get_seq_score(s) for s in list(mdp._encode(mdp.test_data).values())],
                          'novelty_score': mdp_pred_score,
                          'label': [id2label[b] for b in list(mdp.test_data.keys())]})
    # Save Result file
    with open(config.OUTPUT_FILE_PATH, 'wb') as f:
        pickle.dump(result, f)
    print("Done.")
    
if __name__ == '__main__':
    
    main()
