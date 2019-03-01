import sys
import numpy as np

''' tiny RL environment for the door opening problem '''
''' note that size of state space is 2^n             '''
class Env:
  def __init__(self,n):            # n = number of doors
    self.n = n
  def reset(self):
    self.doors = np.zeros(self.n)  # all doors are closed
    self.count = 0
    return self.n_state()
  def step(self,action):
    self.count += 1
    self.doors[action] = 1 if self.doors[action]==0 else 0
    return self.n_state(),-1,sum(self.doors)==self.n or self.count>self.n**3, self.count
  def n_state(self):
    n = 0
    for i,d in enumerate(self.doors): n+= d*(2**i)
    return int(n)
  def sample(self):
    return np.random.randint(self.n)

''' train to open all the doors, using: python doors n epsodes '''
''' to try four doors over 1000 episodes, it's: python 4 1000  '''
n = int(sys.argv[1])
env = Env(n)
q = np.zeros((2**n,n))  # generally: don't use zeros! see post...
explore = 0.5
for n in range(int(sys.argv[2])):
  state = env.reset()
  done = False
  while not done:
    if np.random.random() < explore:
      action = env.sample()                   # explore
    else:
      action = np.argmax(q[state])            # exploit
    obs,reward,done,count = env.step(action)
    q[state][action] = reward+np.max(q[obs])  # only store most recent result
    state = obs
  explore *= 0.5
print(q)
