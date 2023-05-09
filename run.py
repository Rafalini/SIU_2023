import random

from src.turtlesim.turtlesim_env_single import TurtlesimEnvSingle


env = TurtlesimEnvSingle()
env.setup('data/scenario.csv', agent_cnt=1)
agents = env.reset()
tname = list(agents.keys())[0]

# 10 kroków 1 żółwia z losowego segmentu, z losową prędkością <0,2;1> i skrętem <-0,3;0,3>
for i in range(10):
	env.step({tname: (random.uniform(.2, 1), random.uniform(-.3, .3))}, realtime=True)
