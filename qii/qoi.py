# -*- coding: utf-8 -*-
"""
Author: Ho Xuan Vinh
Contact: hxvinh.hcmus@gmail.com
"""

import numpy as np

"""
QuantityOfInterest is a required input to run QuantitativeInputInfluence,
which defines how to compute the quantity of interest with respect to a data
point x_0. By default, we have 2 methods implemented.
"""
class QuantityOfInterest(object):
    def __init__(self):
        self._qoi_dict = {'label_unchanged_ratio': self.compute_label_unchanged_ratio,
                           'label_changed_ratio': self.compute_label_changed_ratio,
                           }
        self._method = self.compute_label_unchanged_ratio
        
    def compute_qoi(self, *args):
        """
        Compute quantity of interest with selected method_name.
        
        @params:
            - *args: all required arguments to run the method.
        @returns:
            - qoi_val: real value has value in range [0., 1.]
        """
        qoi_val = self._method(*args)
        
        return qoi_val
        
    def add_qoi(self, method_name, method):
        self._qoi_dict[method_name] = method
    
    def set_qoi(self, method_name):
        try:
            self._method = self._qoi_dict[method_name]
        except KeyError:
            print ('This method is not added yet.')
        
    def compute_label_unchanged_ratio(self, x_0, x_pool, feature_idxs, predictor):
        """
        Compute the ratio of unchanged predicted label when replace features
        in feature_ind with their empirical distribution, with respect to 
        data point x_0 and its predicted label.
        
        @params:
            - x_0: point of interest, following the format of predictor's input.
            - x_pool: sampled data points from the dataset used for training
            the predictor, following the format of predictor's input.
            - feature_idxs: list of index of features to be replaced with their
            empirical distribution, 1-D list.
            - predictor: a classification model, child class of QII_Predictor.
        @returns:
            - label_unchanged_ratio: real value has value in range [0., 1.]
        """
        
        # 1. determine the predicted label of x_0
        y_0 = predictor.predict(x_0)
        
        # 2. sample emperical data and replace them with values of x_0
        x_sampled = self._sample_empirical_data(x_0, x_pool, feature_idxs)

        # 3. make prediction on x_0 and compute label_unchanged_ratio
        prediction = predictor.predict(x_sampled)
        label_unchanged_ratio = np.mean(np.equal(prediction, y_0))
        
        return label_unchanged_ratio
    
    def compute_label_changed_ratio(self, x_0, x_pool, feature_idxs, predictor):
        """
        Compute the ratio of changed predicted label when replace features
        in feature_ind with their empirical distribution, with respect to 
        data point x_0 and its predicted label.
        
        @params:
            - x_0: point of interest, following the format of predictor's input.
            - x_pool: sampled data points from the dataset used for training
            the predictor, following the format of predictor's input.
            - feature_idxs: list of index of features to be replaced with their
            empirical distribution, 1-D list.
            - predictor: a classification model, child class of QII_Predictor.
        @returns:
            - label_changed_ratio: real value has value in range [0., 1.]
        """
        
        label_changed_ratio = 1. - self.compute_label_unchanged_ratio(x_0, x_pool,
                                                                      feature_idxs,
                                                                      predictor)
        
        return label_changed_ratio
    
    def _sample_empirical_data(self, x_0, x_pool, feature_idxs):
        """
        @params:
            - x_0: 2-D numpy array with shape (1, n_features).
            - x_pool: 2-D numpy array with shape (n_data_points, n_features).
            - feature_idxs: list of integers, values are in range [0, n_features-1].
        @returns:
            - x_sampled: x_pool is replaced with empirical distribution, 
            shape = (n_data_points, n_features).
        """
        pool_size = x_pool.shape[0]
        x_sampled = np.repeat(x_0, pool_size, axis=0)
        
        for feature_idx in feature_idxs:
            x_sampled[..., feature_idx] = x_pool[..., feature_idx]
            
        return x_sampled
