import gym
import numpy as np
import matplotlib as mpl # special magic for repl.it
import matplotlib.pyplot as plt

alpha = 0.02
epsilon = 1.00
eps_min = 0.02
decay = 0.99995
iters = 50000

env=gym.make('MountainCar-v0')
n = 12 # arbitrary number of boundaries to discretize observation
q = np.random.uniform(low=-1,high=1,size=(n+1,n+1,env.action_space.n))
limits = (env.observation_space.high-env.observation_space.low, env.observation_space.low)

def discrete(obs,n):
  return np.round(((obs-limits[1])/limits[0])*n).astype(int)

max_positions = []
recent = np.zeros((100))                    # buffer for graphing results

for i in range(iters):
  total = 0
  max_pos = -float('inf')
  ds = discrete(env.reset(),n)              # discrete initial state (m,n)
  done = False
  while not done:
    if np.random.random()<epsilon:
      action = env.action_space.sample()    # explore
    else:
      action = np.argmax(q[ds[0],ds[1]])    # exploit
    obs,reward,done,_ = env.step(action)
    max_pos = max(max_pos,obs[0])           # progress... for graph at end
    do = discrete(obs,n)                    # discrete observation (m,n)
    q[ds[0],ds[1],action] = (1-alpha)*q[ds[0],ds[1],action] + alpha*(reward+np.max(q[do[0],do[1]]))
    ds = do # current observation becomes next step's initial state
    total += reward
  epsilon = max(epsilon*decay,eps_min)      # decay epsilon

  recent[i%100] = max_pos                   # results for graph
  if i>=100: max_positions.append(np.mean(recent))
  if i%100==0: print('episode {:d}, max pos {:3.2f}, avg max(last 100) {:3.2f}'.
    format(i,max_pos,np.mean(recent)))

plt.plot(max_positions)
plt.title('alpha = ' + str(alpha) + ' n = ' + str(n) + ' episodes = ' + str(iters))
plt.savefig('conventional-q.png')
plt.show()
