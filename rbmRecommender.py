"""

Copyright (c) 2014 Edgar A. Margffoy T.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

import os
import numpy as np
import scipy.io as sio


def clear():
   '''
   Clear terminal screen
   '''
   os.system('clear')

def flattenMatrix(matrix):
   matrix = matrix.flatten(1)
   return matrix.reshape(len(matrix), 1)

class RBM:
   def __init__(self, hiddenSize, data, weights):
      self.inputs = data
      #self.visibleSize = data.shape[0]
      self.hiddenSize = hiddenSize
      self.W = weights
      self.randomness_source = np.random.rand(1, 7000000)
      self.sampleBinaryCalls = False

   def logistic(self, z):
      return 1.0/(1.0+np.exp(-z))
      
   def extractBatch(self,  num_user):
      #minibatch = self.inputs[:, start_i : start_i + n_cases]
      return self.inputs[num_user]
      
   def randomize(self, size, seed):
      start_i = np.mod(np.round(seed), np.round(self.randomness_source.shape[1] / 10)) + 1
      if start_i + np.prod(size) >= self.randomness_source.shape[1] + 1:
         raise Exception("randomize failed to generate an array of that size (too big)")
      rand = self.randomness_source[0, start_i : start_i+np.prod(size)]
      ret = rand.reshape(size, order='F')
      return ret
   
   def sampleBinary(self, probabilities):
      seed = np.sum(flattenMatrix(probabilities))
      binary = 0 + (probabilities > self.randomize(probabilities.shape, seed))
      if self.sampleBinaryCalls:
         print "sampleBinary was called with a matrix of size (%d, %d)" % (probabilities.shape[0], probabilities.shape[1])
      return binary
     
   def visible_to_hiddenProbabilities(self, W, visible_data):
      term = np.dot(W, visible_data)
      hiddenActivation = self.logistic(term)
      return hiddenActivation

   def hidden_to_visibleProbabilities(self, W, hidden_activation):
      term = np.dot(W.T, hidden_activation)
      visibleReconstruction = self.logistic(term)
      return visibleReconstruction
     
   def configurationGradient(self, visibleState, hiddenState):
      gradient = np.dot(visibleState, hiddenState.T)/(np.float32(visibleState.shape[1]));
      return gradient
      
   def CDN(self, W, data, n):
      visibleData = data
      for i in range(0, n):
         visibleData = self.sampleBinary(visibleData)
         hiddenProbabilities = self.visible_to_hiddenProbabilities(W, visibleData)
         hiddenBinary = self.sampleBinary(hiddenProbabilities)
         if not i:
            gradient1 = self.configurationGradient(visibleData, hiddenBinary)
         visibleProbabilities = self.hidden_to_visibleProbabilities(W, hiddenBinary)
         visibleBinary = self.sampleBinary(visibleProbabilities)
         hiddenProbabilities = self.visible_to_hiddenProbabilities(W, visibleBinary)
         if i == n-1:
            gradient2 = self.configurationGradient(visibleBinary, hiddenProbabilities)
            break
         hiddenBinary = self.sampleBinary(hiddenProbabilities)
         visibleData = self.hidden_to_visibleProbabilities(W, hiddenBinary)
      gradient = gradient1 - gradient2
      return gradient.T
      
   def optimize(self, n_iterations, n):
      weights = self.W.copy()
      learning_rate = 0.01
      momentum_speed = np.zeros(self.W.shape)
      num_user = 0
      for iteration_number in range(0, n_iterations+1):
          learning_rate = 0.01
          clear()
          num_user = np.mod(iteration_number+1, len(self.inputs))
          if num_user == 0:
             n += 1 
             num_user = 1
          mini_batch = self.extractBatch(num_user)
          print 'Iteration %d | User # %d - Movies: %d | CD%d\n' % (iteration_number, num_user+1, len(mini_batch), n)
          learning_rate = learning_rate/float(len(mini_batch))
          itemsRat = [i.keys()[0]-1 for i in mini_batch]
          ratings = [i.values()[0]-1 for i in mini_batch]   
          W_batch = weights[:, itemsRat].copy()
          init_Wb = W_batch.copy()
          momentum_batch = momentum_speed[:, itemsRat].copy()
          labels = np.zeros((5, len(itemsRat)))
          labels[ratings, range(0, len(itemsRat))] = 1
          gradient = self.CDN(W_batch, labels.T, n)
          momentum_batch = 0.9 * momentum_batch + gradient
          W_batch = W_batch + momentum_batch * learning_rate
          weights[:, itemsRat] = W_batch
          momentum_speed[:, itemsRat] = momentum_batch
 
      self.W = weights 
      
   def train(self, iterations, n):
      self.optimize(iterations, n)
      
         
          
            
