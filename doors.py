import sys
import numpy as np

class Env:
  def __init__(self,n):
    self.n = n
  def reset(self):
    self.doors = np.zeros(self.n) # all doors are closed
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

# run from commmand line using something like: python doors 4 1000
n = int(sys.argv[1])
env = Env(n)
q = np.zeros((2**n,n))  # generally: don't use zeros! see post
epsilon = 0.5
for n in range(int(sys.argv[2])):
  state = env.reset()
  done = False
  total_reward = 0
  while not done:
    if np.random.random() < epsilon:
      action = env.sample()
    else:
      action = np.argmax(q[state])
    obs,reward,done,count = env.step(action)
    total_reward += reward
    q[state][action] = reward+np.max(q[obs])
    state = obs
  epsilon *= 0.5
print(q)

