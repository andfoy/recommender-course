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

   def visibleToHiddenProbabilities(self, hidden_state):
     term = np.dot(self.W.T, hidden_state)
     return logistic(self, term)
     
   def hiddenToVisibleProbabilities(self, visible_state):
     term = np.dot(self.W, visible_state)
     return logistic(self, term)

   def configurationGradient(visible_state, hidden_state):
     dG_by_W = np.dot(visible_state, hidden_state.T)/(np.float32(visible_state.shape[1]));
     return dG_by_W
   
   

