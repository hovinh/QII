# -*- coding: utf-8 -*-
"""
Author: Ho Xuan Vinh
Contact: hxvinh.hcmus@gmail.com
"""

"""
QIIPredictor is a required input to run QuantitativeInputInfluence,
which is a wrapper of the model to be evaluated.
"""
class QIIPredictor(object):
    def __init__(self, predictor):
        self._predictor = predictor
        
    def predict(self, x):
        raise NotImplementedError("You need to implement predict() method.")
        
