import numpy as np
import scipy.io as sio

def flattenMatrix(matrix):
   matrix = matrix.flatten(1)
   return matrix.reshape(len(matrix), 1)

class RBM:
   def __init__(self, visibleSize, hiddenSize, data, weights):
      self.inputs = data
      self.visibleSize = visibleSize
      self.hiddenSize = hiddenSize
      self.W = weights
      self.randomness_source = np.random.rand(1, 7000000)

   def logistic(z):
      return 1.0/(1.0+np.exp(-z))
      
   def randomize(self, size, seed):
      start_i = np.mod(np.round(seed), np.round(randomness_source.shape[1] / 10)) + 1
      if start_i + np.prod(size) >= randomness_source.shape[1] + 1:
         raise Exception("randomize failed to generate an array of that size (too big)")
      rand = randomness_source[0, start_i : start_i+np.prod(size)]
      ret = rand.reshape(size, order='F')
      return ret
   
   def sampleBinary(self, probabilities):
      seed = np.sum(flattenMatrix(probabilities))
      binary = 0 + (probabilities > randomize(self, probabilities.shape, seed))
      return binary
     
   def visible_to_hiddenProbabilities(self, visible_data):
      term = np.dot(self.W, visible_data)
      hiddenActivation = logistic(self, term)
      return hiddenActivation

   def hidden_to_visibleProbabilities(self, hidden_activation):
      term = np.dot(self.W.T, hidden_activation)
      visibleReconstruction = logistic(self, term)
      return visibleReconstruction
     
   def configurationGradient(self, visibleState, hiddenState)
      gradient = np.dot(visible_state, hidden_state.T)/(np.float32(visible_state.shape[1]));
      return gradient
      
   def CDN(self, n):
      inputData = self.inputs
      for i in range(0, n):
         visibleData = sampleBinary(self, visibleData)
         hiddenProbabilities = visible_to_hiddenProbabilities(self, visible_data)
         hiddenBinary = sampleBinary(self, hiddenProbabilities)
         if not i:
            gradient1 = configurationGradient(self, visibleData, hiddenBinary)
         gradient1N = configurationGradient(self, visibleData, hiddenBinary)
         
            
