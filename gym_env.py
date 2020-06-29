# import gym
# env = gym.make('MountainCar-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     print("Episode {:d} ".format(i_episode))
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = 1  # env.action_space.sample()
# #        print(action)
# #        print(t)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# import gym
# env = gym.make('Pendulum-v0')
# print(env.action_space.shape[0])  # Jointeffort  -2 ~ +2
# print(env.observation_space.shape[0])  # cos(theta)  sin(theta)  theta_dot
# print(env.observation_space.high)
# print(env.observation_space.low)

# 'LunarLander-v2'
import gym
env = gym.make('Pendulum-v0')
for i_episode in range(2):
    observation = env.reset()
    print("Episode {:d} ".format(i_episode))
    for t in range(10000):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        print(action)
        # print(t)
        observation, reward, done, info = env.step(action)
        print(reward,'reward',done, 'done',action,'action')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
