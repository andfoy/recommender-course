import rbm
import scipy
import cpickle
import numpy as np
 
'''
This implementation is based on the MovieLens 100k dataset,
available at http://grouplens.org/datasets/movielens/
'''
 
 
numUsers = 1682
hidSize = 100
K = 5
 
with open('u.data', 'r') as data:
  rat = data.readlines()

users = {}
 
for i in rat:
  r = i.split('\t')[0:-1]
  if int(r[0]) in users.keys():
    user = users[int(r[0])]
    user.append({int(r[1]):int(r[2])})
    users[int(r[0])] = user
  else:
    users[int(r[0])] = [{int(r[1]):int(r[2])}]
 
for j in users.keys():
  user = users[j]
  newUser = []
  movies = [i.keys()[0] for i in user]
  movies.sort()
  for m in movies:
    for mov in user:
      if mov.keys()[0] == m:
         newUser.append(mov)
         users[j] = newUser
 
with open('userRatings', 'wb') as fp:
  cPickle.dump(users)
 
'''
Initialize RBM parameters
'''
 
W = np.random.normal(0, 0.1, numUsers*hidSize)
W = W.reshape(numUsers, hidSize)
 
for i in user.keys():
   moviesRat = [j.keys()[0]-1 for j in user[i]]
   ratings = [j.values()[0]-1 for j in user[i]]
   W_U = W[moviesRat, :]
   labels = np.zeros((K, len(moviesRat)))
   labels[ratings, moviesRat] = 1
   rbm = RBM()
