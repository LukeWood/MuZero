from game.breakout import Breakout
env = Breakout(0.99)
print(env.env.observation_space)
state =  env.env.reset()
print(state.shape)
