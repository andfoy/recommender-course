import os
import numpy as np
import scipy.io as sio

def clear():
   os.system('clear')

def flattenMatrix(matrix):
   matrix = matrix.flatten(1)
   return matrix.reshape(len(matrix), 1)

class RBM:
   def __init__(self, visibleSize, hiddenSize, data, weights):
      self.inputs = data
      self.visibleSize = data.shape[0]
      self.hiddenSize = hiddenSize
      self.W = weights
      self.randomness_source = np.random.rand(1, 7000000)

   def logistic(z):
      return 1.0/(1.0+np.exp(-z))
      
   def extractBatch(self,  start_i, n_cases):
      minibatch = self.inputs[:, start_i : start_i + n_cases]
      return minibatch
      
   def randomize(self, size, seed):
      start_i = np.mod(np.round(seed), np.round(self.randomness_source.shape[1] / 10)) + 1
      if start_i + np.prod(size) >= self.randomness_source.shape[1] + 1:
         raise Exception("randomize failed to generate an array of that size (too big)")
      rand = self.randomness_source[0, start_i : start_i+np.prod(size)]
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
      
   def CDN(self, data, n):
      inputData = data
      for i in range(0, n):
         visibleData = sampleBinary(self, visibleData)
         hiddenProbabilities = visible_to_hiddenProbabilities(self, visible_data)
         hiddenBinary = sampleBinary(self, hiddenProbabilities)
         if not i:
            gradient1 = configurationGradient(self, visibleData, hiddenBinary)
         visibleProbabilities = hidden_to_visibleProbabilities(self, hiddenBinary)
         visibleBinary = sampleBinary(self, visibleProbabilities)
         hiddenProbabilities = visible_to_hiddenProbabilities(self, visibleBinary)
         if i == n-1:
            gradient2 = configurationGradient(self, visibleBinary, hiddenProbabilties)
            break
         hiddenBinary = sampleBinary(self, hiddenProbabilities)
         visibleData = hidden_to_visibleProbabilities(self, hiddenBinary)
      gradient = gradient1 - gradient2
      return gradient.T
      
   def optimize(self, model_shape, learning_rate, n_iterations, n):
      momentum_speed = np.zeros(model_shape)
      mini_batch_size = 100;
      start_of_next_mini_batch = 0;
      for iteration_number in range(0, n_iterations+1):
          clear()
          print 'Iteration %d | Batch # %d\n' % (iteration_number, start_of_next_mini_batch)
          mini_batch = extractBatch(self, start_of_next_mini_batch, mini_batch_size)
          start_of_next_mini_batch = np.mod(start_of_next_mini_batch + mini_batch_size, training_data.shape[1])
          gradient = CDN(self, model, mini_batch)
          momentum_speed = 0.9 * momentum_speed + gradient
          self.W = self.W + momentum_speed * learning_rate
      return model
      
   def train(self, learning_rate, iterations, n):
      model_shape = (self.hiddenSize, self.visibleSize)
      optimize(self, model_shape, learning_rate, iterations, n)
      
         
          
            
