import gym
import numpy as np
import matplotlib as mpl # special magic for repl.it
import matplotlib.pyplot as plt

##############################################################
#                                                            #
# OpenAI MountainCar-v0 (Moore 1990) -- argmax() only        #
#                                                            #
##############################################################

#alpha = 0.02     # no alpha term -- learning rate is 1.0
#gamma = 0.90     # don't discount future rewards
#explore = 1.00   # initial exploration rate
#exp_min = 0.005  # minimum exploration rate
#decay = 0.9999   # episodic decay of exploration rate
iters = 50000     # total episodes
render = False

env=gym.make('MountainCar-v0')
n = 12 # arbitrary number of boundaries to discretize observation
q = np.zeros((n+1,n+1,env.action_space.n))
limits = (env.observation_space.high-env.observation_space.low, env.observation_space.low)

# convert continous observations evenly into discrete observations
def discrete(obs,n):
    return np.round(((obs-limits[1])/limits[0])*n).astype(int)

max_positions = []                        
recent = np.zeros((100))                  

for i in range(iters):
    max_pos = -float('inf')
    ds = discrete(env.reset(),n)      
    done = False
    while not done:
        action = np.argmax(q[ds[0],ds[1]])      # argmax only!
        obs,reward,done,_ = env.step(action)
        max_pos = max(max_pos,obs[0])         
        do = discrete(obs,n)                    
        q[ds[0],ds[1],action] = reward+np.max(q[do[0],do[1]])
        ds = do                               
        if render and i%1000==0: env.render()

    ''' just reporting results from here on '''
    recent[i%100] = max_pos
    if i>=100: max_positions.append(np.mean(recent))
    if i%1000==0: print('episode {:d}, max pos {:3.2f}, avg max(last 100) {:3.2f}'.
                  format(i,max_pos,np.mean(recent)))

plt.plot(max_positions)
plt.title('argmax only n {:d} episodes {:d}'.format(n,iters))
plt.savefig('argmax-only.png')
plt.show()
