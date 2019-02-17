# -*- coding: utf-8 -*-
"""
Author: Ho Xuan Vinh
Contact: hxvinh.hcmus@gmail.com
"""

import sys
import math
import numpy as np
import itertools

"""
QuantitativeInputInfluence - the transparency method takes in a QIIPredictor
and QuantityOfInterest, and produce the influence score for each feature in 
a data point x_0 with respect to its dataset it belongs to.
"""
class QII(object):
    def __init__(self, data, n_features, quantity_of_interest):
        '''
        @params:
            - data: the dataset is used for evaluation.
            - n_features: total number of features in input, an integer.
            - quantity_of_interest: an instance of QuantityOfInstance class/child class.
        '''
        self._data = data
        self._n_features = n_features
        self._feature_idxs = [i for i in range(self._n_features)]
        self._qoi = quantity_of_interest
        
    def compute(self, x_0, predictor, show_approx=False, evaluated_features=None,
                data_exhaustive=False, feature_exhaustive=False,
                pool=None, pool_size=600, n_samplings=600, method='shapley'):
        """
        @params:
            - x_0: data point to be evaluated.
            - predictor: an instance of QIIPredictor class/child class. 
            - show_approx: show summation of unnormalized influence scores in
            progress bar or not, a boolean.
            - evaluated_features: integer list of feature want to evaluate, starting 
            from 0 and ending at <n_features-1>.
            - data_exhaustive: take the whole dataset as sampled pool or not,
            a boolean.
            - feature_exhaustive: compute quantity of interest for all possible 
            combinations of features or not, a boolean.
            - pool: index list of specific instances, a 1-D integer list.
            - pool_size: the number of instances to be sampled from, an integer.
            - n_samplings: number of samplings.
            - methods: a string, "shapley" or "banzhaf".
        @returns:
            - influence_scores: dictionary contains influence scores for each feature
            in evaluated_features.
        """
        
        if method not in ['shapley', 'banzhaf']:
            print ('Available options for method: "shapley", "banzhaf". Default: "shapley"')
        else:
           
            # 1. Set up variables before compute Shapley/Banzhaf value
            self._x_0 = x_0
            self._predictor = predictor
            self._show_approx = show_approx
            self._determine_params(evaluated_features, data_exhaustive, 
                                   feature_exhaustive, pool, pool_size,
                                   n_samplings, method)
    
            # 2. Compute influence scores
            if (method == 'shapley'):
                influence_scores = self.compute_shapley()
            else:
                influence_scores = self.compute_banzhaf()
                
            return influence_scores

    def compute_unary_qii(self, si, S):

        qoi_S = self._qoi.compute_qoi(self._x_0,
                                      self._x_pool,
                                      S,
                                      self._predictor)
        S_si = S + [si]
        qoi_S_si = self._qoi.compute_qoi(self._x_0,
                                         self._x_pool,
                                         S_si,
                                         self._predictor)
        
        return qoi_S - qoi_S_si
        
    def compute_shapley(self):
    
        shapley = dict.fromkeys(self._evaluated_features, 0.)
        n_features = len(self._evaluated_features)
            
        # Calculate Shapley value...
        if (self._feature_exhaustive is True):
            # exactly
            permutations = list(itertools.permutations(self._evaluated_features))
            for count_samplings, sample in enumerate(permutations):
                
                perm = sample
                
                for i in range(0, n_features):
                    # Choose the set of feature standing before i in the permutation
                    si = perm[i]
                    S = [perm[j] for j in range(0, i)]
                    shapley[si] = shapley[si] + self.compute_unary_qii(si, S)
                    
                self._update_progress(iter_idx=count_samplings+1,
                                      n_iters=self._n_samplings,
                                      show_approx=self._show_approx,
                                      influence_scores=shapley)
                
        else:
            # or approximately
            for count_samplings in range(0, self._n_samplings):
                
                perm = np.random.permutation(self._evaluated_features)
                
                for i in range(0, n_features):
                    si = perm[i]
                    S = [perm[j] for j in range(0, i)]
                    shapley[si] = shapley[si] + self.compute_unary_qii(si, S)
                
                self._update_progress(iter_idx=count_samplings+1,
                                      n_iters=self._n_samplings,
                                      show_approx=self._show_approx,
                                      influence_scores=shapley)

        for feature in shapley.keys():
            shapley[feature] = shapley[feature]/self._n_samplings
        
        return shapley
    
    def compute_banzhaf(self):
        
        banzhaf = dict.fromkeys(self._evaluated_features, 0)
        def get_all_combinations(feature_idxs):
            n_features = len(feature_idxs)
            combinations = []
            for subset_size in range(1, n_features+1):
                combination_subset_i = list(itertools.combinations(feature_idxs, subset_size))
                combinations = combinations + combination_subset_i
            return combinations
            
        # Calculate Banzhaf value...
        if (self._feature_exhaustive is True):
            # exactly
            combinations = get_all_combinations(self._evaluated_features)
            for count_samplings, sample in enumerate(combinations):
                
                subset = sample
                for si in subset:
                    # Chooses the set of feature in evaluated_features, excluding i
                    S = [j for j in subset if j != si]
                    banzhaf[si] = banzhaf[si] + self.compute_unary_qii(si, S)
                    
                self._update_progress(iter_idx=count_samplings+1,
                                      n_iters=self._n_samplings,
                                      show_approx=self._show_approx,
                                      influence_scores=banzhaf)
                
        else:
            # or approximately
            n_features = len(self._evaluated_features)
            for count_samplings in range(0, self._n_samplings):
                    
                r = np.random.ranf(n_features)
                subset = [i for i in range(n_features) if r[i] > 0.5]
                
                for si in subset:
                    S = [j for j in subset if j != si]
                    banzhaf[si] = banzhaf[si] + self.compute_unary_qii(si, S)
                    
                self._update_progress(iter_idx=count_samplings+1,
                                      n_iters=self._n_samplings,
                                      show_approx=self._show_approx,
                                      influence_scores=banzhaf)
                
        for feature in banzhaf.keys():
            banzhaf[feature] = banzhaf[feature]/self._n_samplings
            
        return banzhaf
    
    
    def _update_progress(self, iter_idx, n_iters, show_approx=False, influence_scores=None):
        """
        Source: https://stackoverflow.com/questions/3160699/python-progress-bar
        
        @params: 
            - iter_idx: the index of current iteration , an integer.
            - n_iters: total number of iterations, an integer.
            - show_approx: whether to show summation of unnormalized influence 
            scores or not, useful for debugging, a boolean.
            - influence_scores: unnormalized influence scores, used when 
            show_approx=True, a dictionary.
        """
            
        progress = float(iter_idx) / n_iters
        barLength = 20 
        status = ""

        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"

        
        block = int(round(barLength*progress))        
        if (show_approx is True):
            inf = sum([influence_scores[i] for i in influence_scores])         
            text = '\rPercent: [{0}] {1}% Count: {2} Approx: {3:.2f} | {4}'.format( 
                    "#"*block + "-"*(barLength-block), int(progress*100),
                    iter_idx,
                    inf,
                    status,
                    )
        else:
            text = '\rPercent: [{0}] {1}% Count: {2} | {3}'.format( 
                    "#"*block + "-"*(barLength-block), int(progress*100),
                    iter_idx,
                    status,
                    )
            
        sys.stdout.write(text)
        sys.stdout.flush()
       
    def _determine_params(self, evaluated_features, data_exhaustive, 
                          feature_exhaustive, pool, pool_size,
                          n_samplings, method):
        """
        Set up following variables:
            + self._evaluated_features
            + self._n_samplings
            + self._x_pool
        @params: check argument list of compute()
        """
        
        if (pool is not None) and (data_exhaustive is True):
            raise ValueError('pool=None or data_exhaustive=False')
        
        # determine features to be evaluated
        if (evaluated_features is None):
            self._evaluated_features = self._feature_idxs
                
        # determine the distribution to be drawn values from
        if (pool is not None):
            self._x_pool = np.copy(pool)
        else:
            if (data_exhaustive is True):
                self._x_pool = np.copy(self._data)
            else:
                pool_idxs = np.random.randint(0, self._data.shape[0], pool_size)
                self._x_pool = np.copy(self._data[pool_idxs])
                
        # determine the number of samplings
        n_features = len(self._evaluated_features)
        if (feature_exhaustive is True):
            if (method == 'shapley'):
                self._n_samplings = math.factorial(n_features) 
            else:
                self._n_samplings = 2**n_features - 1
        else:
            if (method == 'shapley'):
                self._n_samplings = min(n_samplings, math.factorial(n_features))
            else:
                self._n_samplings = min(n_samplings, 2**n_features - 1)

        self._data_exhaustive = data_exhaustive
        self._feature_exhaustive = feature_exhaustive