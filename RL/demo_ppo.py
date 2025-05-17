import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from torch.distributions import Categorical


# 定义神经网络模型
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.gamma = 0.99  # 折扣因子
        self.eps_clip = 0.2  # 剪裁系数
        self.gae_lambda = 0.95
        self.epochs = 4  # 更新轮次
        self.batch_size = 64  # 批大小

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮次更新
        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                batch = slice(idx, idx + self.batch_size)

                # 获取新策略的概率和状态价值
                new_probs, state_values = self.model(states[batch])
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()

                # 计算概率比率
                ratios = torch.exp(new_log_probs - old_log_probs[batch])

                # 计算策略损失（含剪裁）
                surr1 = ratios * advantages[batch]
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages[batch]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = (
                    0.5 * (returns[batch] - state_values.squeeze()).pow(2).mean()
                )

                # 总损失（含熵奖励）
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs, value = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


# 训练函数
def train_ppo(env_name="CartPole-v1", max_episodes=300):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim)

    for episode in range(max_episodes):
        # 修复：处理新版gym的返回值格式
        state, _ = env.reset()
        done = False
        rewards = []
        states, actions, log_probs, values = [], [], [], []

        # 收集数据
        while not done:
            action, log_prob, value = agent.get_action(state)
            # 修复：处理新版gym的step返回值格式
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # 计算回报和优势
        returns = []
        advantages = []
        R = 0
        for r in reversed(rewards):
            R = r + agent.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)

        # 修复：确保维度匹配
        # 方法1：截断returns使其与values[:-1]匹配
        advantages = returns[:-1] - values[:-1]

        # 或者方法2：使用GAE计算优势（更推荐）
        # gae = 0
        # advantages = []
        # for i in reversed(range(len(rewards)-1)):
        #     delta = rewards[i] + agent.gamma * values[i+1] - values[i]
        #     gae = delta + agent.gamma * agent.gae_lambda * gae
        #     advantages.insert(0, gae)
        # advantages = torch.FloatTensor(advantages)

        # 更新模型时也需要相应调整
        agent.update(
            states[:-1], actions[:-1], log_probs[:-1], returns[:-1], advantages
        )

        # 打印训练进度
        total_reward = sum(rewards)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

        # 提前终止条件（可选）
        if total_reward >= 475:
            print("Solved!")
            break

    env.close()


if __name__ == "__main__":
    train_ppo()
