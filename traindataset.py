'''
Created on Oct 26, 2017

@author: kwibu
'''
import numpy as np
        
class dataset:
    def __init__(self, data):
        self.data = data
        self.n_data = data.shape[0]
            
    def next_batch(self, n_batches):
        indices = np.random.randint(self.n_data, size=n_batches)
        return self.data[indices]
    