import gym
import pybulletgym
import torch
from model import PPO
from torch.distributions import Normal
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import argparse

class Agent:
    def __init__(self, environment, device):
        self.env_id = environment
        self.device = device
        self.writter = SummaryWriter()
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def make_env(self):
        def thunk():
            env = gym.make(self.env_id)
            return env
        return thunk

    def calculate_gae(self,next_value,gamma, lmda):
        values =  self.values + [next_value]
        gae= 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + gamma * lmda * self.masks[step] * gae
            returns.append(gae + values[step])
        return list(reversed(returns))

    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def sample_batch(self, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)

        for _ in range(batch_size // args.mini_batch):
            rand_ids = np.random.randint(0, batch_size , args.mini_batch)
            yield states[rand_ids, :], actions[rand_ids, :],log_probs[rand_ids, :], \
                    returns[rand_ids, :], advantages[rand_ids, :]


    def learn(self):
        envs = gym.vector.SyncVectorEnv([self.make_env() for i in range(args.n_workers)])
        num_inputs = envs.observation_space.shape[1]
        num_outputs = envs.action_space[0].shape[0]

        model = PPO(num_inputs, num_outputs).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr = args.lr)

        frame_idx = 0
        train_epoch = 0
        best_reward = None

        state = envs.reset()
        early_stop = None

        for _ in range(args.ppo_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(torch.tensor(reward, dtype = torch.float32).unsqueeze(1).to(self.device))
            self.masks.append(torch.tensor(1 - done).unsqueeze(1).to(self.device))
            self.states.append(state)
            self.actions.append(action)

            state = next_state
            frame_idx += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = self.calculate_gae(next_value, args.gamma, args.lmda)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        values = torch.cat(self.values).detach()
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        advantage = returns - values
        advantage = self.normalize(advantage)

        count_steps = 0
        sum_returns = 0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        for _ in range(args.epochs):
            for state, action, old_log_probs, retn, adv in self.sample_batch(states, actions, log_probs, returns, advantage):

                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0- args.epsilon, 1.0 + args.epsilon) * adv

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (retn  - value).pow(2).mean()

                total_loss = args.c1  * critic_loss + actor_loss - args.c2 * entropy

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


        

        
if __name__ == "__main__":
    
    HIDDEN_SIZE         = 256
    LEARNING_RATE       = 1e-4
    GAMMA               = 0.99
    GAE_LAMBDA          = 0.95
    PPO_EPSILON         = 0.2
    CRITIC_DISCOUNT     = 0.5
    ENTROPY_BETA        = 0.001
    PPO_STEPS           = 256
    MINI_BATCH_SIZE     = 64
    PPO_EPOCHS          = 10
    TEST_EPOCHS         = 10
    NUM_TESTS           = 10
    TARGET_REWARD       = 2500

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help = "OpenAI gym environment", default = "HalfCheetahPyBulletEnv-v0", type = str)
    parser.add_argument("-n_workers", help = "Number of environments", default = 8, type = int)
    parser.add_argument("-mini_batch", help = "Size of mini batch to sample", default = 64, type = int)
    parser.add_argument("-lr", help = "Model learning rate", default = 1e-4, type = float)
    parser.add_argument("-gamma", help = "return discount factor", default = 0.99, type = float)
    parser.add_argument("-lmda", help = "gae lambda", default = 0.95, type = float)
    parser.add_argument("-epochs", help = "number of updates", default = 10, type = int)
    parser.add_argument("-ppo_steps", help = "Number of steps before update", default = 256, type = int)
    parser.add_argument("-c1", help = "critic discount", default = 0.5, type = float)
    parser.add_argument("-c2", help = "entropy beta", default = 0.001, type = float)
    parser.add_argument("-epsilon", help = "entropy beta", default = 0.02, type = float)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(args.env, device)
    agent.learn()
    
