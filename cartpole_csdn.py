import argparse
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from zhihulogger import Logger

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        # print(action_scores)
        # print(F.softmax(action_scores, dim=1))
        return F.softmax(action_scores, dim=1) # softmax是个非常常用而且比较重要的函数，尤其在多分类的场景中使用广泛。他把一些输入映射为0-1之间的实数，并且归一化保证和为1，因此多分类的概率之和也刚好为1。

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    # print(probs)
    m = Categorical(probs)  # probs是概率，Categorical()按照概率构造一个分布，返回数组的索引
    # print(m)
    action = m.sample()  # 按照概率进行采样，返回数组中的值，也就是Categorical()的值，probs的索引
    # print(action)
    policy.saved_log_probs.append(m.log_prob(action))  # 动作对应概率的log值
    return action.item()


def finish_episode(i_episode):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:  # [::-1] 从最后一个往前数到第一个，也就是倒序排列
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # print(eps)  # 防除零
    # print(returns)
    # print(returns.mean())
    returns = (returns - returns.mean()) / (returns.std() + eps)  # .mean()求平均 ～～ .std()求方差
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)  # loss = logP*V
    # print(policy_loss, '0')
    # print(torch.tensor(policy_loss).sum(), '1')
    optimizer.zero_grad()
    # print(torch.cat(policy_loss), '2')
    policy_loss = torch.cat(policy_loss).sum()
    # print(policy_loss, '3')
    policy_loss.backward()
    optimizer.step()
    '''
    # ========================= Log Start ======================

    step = i_episode
    # (1) Log the scalar values
    info = {'loss': policy_loss.item()}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)
    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in policy.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.detach().numpy(), step)
        logger.histo_summary(tag + '/grad', value.grad.detach().numpy(), step)
    # ========================= Log End ======================
    '''
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    #logger = Logger('./logpend')
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            # print(action)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(i_episode)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:

            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            torch.save(policy.state_dict(), 'hello.pt')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    """
    np.finfo使用方法
        eps是一个很小的非负数
        除法的分母不能为0的,不然会直接跳出显示错误。
        使用eps将可能出现的零用eps来替换，这样不会报错。
    """
    eps = np.finfo(np.float64).eps.item()  # 生成一个接近于0的绝对值很小的数，防止除零
    env = gym.make('CartPole-v1')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    main()

# 结果测试
# if __name__ == '__main__':
#     model = Policy()
#     model.load_state_dict(torch.load('hello.pt'))
#     model.eval()
#
#
#     def select_action(state):
#         state = torch.from_numpy(state).float().unsqueeze(0)
#         probs = model(state)
#         m = Categorical(probs)
#         action = m.sample()
#         return action.item()
#
#
#     env = gym.make('CartPole-v1')
#     t_all = []
#     for i_episode in range(5):
#         observation = env.reset()
#         for t in range(10000):
#             env.render()
#             cp, cv, pa, pv = observation
#             action = select_action(observation)
#             observation, reward, done, info = env.step(action)
#             if done:
#                 print("Episode finished after {} timesteps".format(t+1))
#                 t_all.append(t)
#                 break
#     env.close()
#     print(t_all)
#     print(sum(t_all) / len(t_all))
