import gym
import numpy as np
import matplotlib.pyplot as plt

################################################################################
#                                                                              #
#  Straightforward Q-learning applied to OpenAI MountainCar-v0 (Moore 1990)    #
#                                                                              #
################################################################################

alpha = 0.02      # learning rate
explore = 1.00    # initial exploration rate
exp_min = 0.005   # minimum exploration rate
decay = 0.9999    # episodic decay of exploration rate
iters = 50000     # total episodes

env=gym.make('MountainCar-v0')
n = 40 # arbitrary number of boundaries to discretize observation
q = np.random.uniform(low=-1,high=1,size=(n+1,n+1,env.action_space.n))
limits = (env.observation_space.high-env.observation_space.low, env.observation_space.low)

# convert continous observations evenly into discrete observations
def discrete(obs,n):
    return np.round(((obs-limits[1])/limits[0])*n).astype(int)

max_positions = []                          # buffer for graphing results
recent = np.zeros((100))                    # buffer for computing recent mean

for i in range(iters):
    max_pos = -float('inf')
    ds = discrete(env.reset(),n)            # discrete starting state (m,n)
    done = False
    while not done:
        if np.random.random()<explore:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q[ds[0],ds[1]])  # exploit
        obs,reward,done,_ = env.step(action)
        max_pos = max(max_pos,obs[0])           # progress... for graph at end
        do = discrete(obs,n)                    # discrete observation (m,n)
        q[ds[0],ds[1],action] = (1-alpha)*q[ds[0],ds[1],action] + \
                                alpha*(reward+np.max(q[do[0],do[1]]))
        ds = do                                 # use obs as next starting state
        if i%1000==0: env.render()
    explore = max(explore*decay,exp_min)        # decay explore rate

    ''' just reporting results from here on '''
    recent[i%100] = max_pos
    if i>=100: max_positions.append(np.mean(recent))
    if i%1000==0: print('episode {:d}, max pos {:3.2f}, avg max(last 100) {:3.2f}, explore {:1.5f}'.
                  format(i,max_pos,np.mean(recent),explore))

plt.plot(max_positions)
plt.title('alpha = ' + str(alpha) + ' n = ' + str(n) + ' episodes = ' + str(iters))
plt.show()
