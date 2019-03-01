import gym

################################################################################
#  try to solve mountain car by sampling action space...                       #
#  ...that trick never works                                                   #
################################################################################

print('trying to solve mountain car using random sampling')

env=gym.make('MountainCar-v0')      # use OpenAI MountainCar (discrete actions)
max_reward, min_reward = -float('inf'), float('inf')
for i in range(10000):              # try lots of episodes...
  env.reset()
  total = 0
  done = False
  while not done:
    obs,reward,done,_ = env.step(env.action_space.sample()) # just guess a move
    total += reward
  max_reward = max(total,max_reward)
  min_reward = min(total,min_reward)
  if (i+1)%1000==0: print(i+1,'max, min total rewards', max_reward, min_reward)
  if max_reward>-200: print('Wow! Stumbled across a solution!')

print('are any of the rewards better than -200?')
