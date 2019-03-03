import numpy as np

# change these values to run various cases
n = 4
episodes = 100

# tiny RL environment for the bit-flipping problem
# note that size of state space is 2^n             
class Env:
  def __init__(self,n):            # n = number of doors
    self.n = n
  def reset(self):
    self.bits = np.zeros(self.n)  # all doors are closed
    self.count = 0
    return self.n_state()
  def step(self,action):
    self.count += 1
    self.bits[action] = 1 if self.bits[action]==0 else 0
    return self.n_state(),-1,sum(self.bits)==self.n or self.count>self.n**3, self.count
  def n_state(self):
    n = 0
    for i,d in enumerate(self.bits): n+= d*(2**i)
    return int(n)
  def sample(self):
    return np.random.randint(self.n)

# train to flip all the bits
env = Env(n)
q = np.zeros((2**n,n))  # generally: don't use zeros! see post...
for n in range(episodes):
  state = env.reset()
  done = False
  while not done:
    action = np.argmax(q[state])       # no epsilon/explore... just argmax
    obs,reward,done,count = env.step(action)
    q[state][action] = reward+np.max(q[obs])
    state = obs
    if n==episodes-1: print(env.bits)  # final state transitions of bits
print('\nsimplified Q table')
print(q)

