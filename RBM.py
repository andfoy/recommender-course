import numpy as np
import scipy.io as sio

class RBM:
   def __init__(self, visibleSize, hiddenSize, data, weights):
     self.inputs = data
     self.visibleSize = visibleSize
     self.hiddenSize = hiddenSize
     self.W = weights

   def logistic(z):
     return 1.0/(1.0+np.exp(-z))

   def visibleToHiddenProbabilities(self):
     return 
